import "puppeteer";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import dotenv from "dotenv";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { zodToJsonSchema } from "zod-to-json-schema";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
dotenv.config();
import { ChatMistralAI } from "@langchain/mistralai";
import { MongoClient } from "mongodb";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { Annotation, MemorySaver, StateGraph } from "@langchain/langgraph";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  isAIMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { v4 as uuidv4 } from "uuid";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// -------------------------------------------------------------------------
// MongoDB connection & document loading/chunking
// -------------------------------------------------------------------------
const client = new MongoClient(`${process.env.MONGODB_URI}`);
await client.connect();
const db = client.db("sophyy");

const webLoader = new PuppeteerWebBaseLoader(
  "https://addissoftware.com/making-the-leap-why-switching-to-rtk-query-from-saga-makes-sense/",
  {
    launchOptions: {
      headless: true,
    },
    gotoOptions: {
      waitUntil: "networkidle0",
      timeout: 60000,
    },
    evaluate: async (page) => {
      return await page.evaluate(() => {
        return document.body.innerText;
      });
    },
  }
);

const docs = await webLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1500,
  chunkOverlap: 300,
});
const allSplits = await splitter.splitDocuments(docs);

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0,
  maxRetries: 2,
  apiKey: process.env.GOOGLE_API_KEY, // Ensure you have set the GOOGLE_API_KEY in your .env file
});

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
  taskType: TaskType.RETRIEVAL_DOCUMENT,
});

const pinecone = new PineconeClient({
  apiKey: `${process.env.PINECONE_API_KEY}`,
});

const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX!);

const vectorStore = await MemoryVectorStore.fromDocuments(
  allSplits,
  embeddings
);

await vectorStore.addDocuments(allSplits);

// -------------------------------------------------------------------------
// Graph State Definition (Updated to append messages)
// -------------------------------------------------------------------------
const graphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (prev, next) => prev.concat(next),
    default: () => [],
  }),
  query: Annotation<string>(),
  relevant_documents: Annotation<Document[]>({
    reducer: (prev, next) => prev.concat(next),
    default: () => [],
  }),
});

// -------------------------------------------------------------------------
// Helper: Document Formatter
// -------------------------------------------------------------------------
const formatDocs = (docs: Document[]) => {
  if (!docs || docs.length === 0) {
    return "No documents found";
  }
  return docs
    .map((doc, i) => {
      const title = doc.metadata?.title || "Untitled";
      const source = doc.metadata?.source || "Unknown source";
      return `<doc${i + 1}>:\nTitle: ${title}\nSource: ${source}\nContent: ${
        doc.pageContent
      }\n</doc${i + 1}>\n`;
    })
    .join("\n");
};

// -------------------------------------------------------------------------
// Tool Node: Retrieval Tool (Modified to filter duplicates)
// -------------------------------------------------------------------------
const retrieveSchema = z.object({ query: z.string() });

const retrieve = tool(
  async ({ query }) => {
    try {
      const retrievedDocs = await vectorStore.similaritySearch(query, 5);
      const uniqueDocs: Document[] = [];
      const seenContent = new Set();

      for (const doc of retrievedDocs) {
        const contentSignature = doc.pageContent.substring(0, 100);
        if (!seenContent.has(contentSignature)) {
          seenContent.add(contentSignature);
          uniqueDocs.push(doc);
        }
      }

      const serialized = formatDocs(uniqueDocs);
      // Return a tuple: first the serialized docs, and then the uniqueDocs array.
      return [serialized, uniqueDocs];
    } catch (error) {
      console.error("Error during document retrieval:", error);
      return ["Sorry, I couldn't retrieve relevant documents.", []];
    }
  },
  {
    name: "retrieve",
    description: "Retrieve information related to a query.",
    schema: retrieveSchema,
    responseFormat: "content_and_artifact",
  }
);

// -------------------------------------------------------------------------
// Nodes using Graph State
// -------------------------------------------------------------------------

// 1. retrieveOrRespond: Either use the tool or respond directly
const retrieveOrRespond = async (
  state: typeof graphState.State
): Promise<Partial<typeof graphState.State>> => {
  console.log("---RETRIEVING OR RESPONDING---");
  const llmWithTools = llm.bindTools([retrieve]);

  // Filter human/system messages
  const messages = state.messages.filter(
    (message) =>
      message instanceof HumanMessage || message instanceof SystemMessage
  );

  const systemMessage = new SystemMessage(
    "When the user asks about technical topics, use the retrieve tool to find relevant information before responding."
  );

  const response = await llmWithTools.invoke([systemMessage, ...messages]);
  return { messages: [response] };
};

// 2. tools: The tool node that runs the retrieval
const tools = new ToolNode([retrieve]);

// 3. checkDocRelevancy: Assess retrieved doc relevance (similar to gradeDocuments)
const checkDocRelevancy = async (
  state: typeof graphState.State
): Promise<Partial<typeof graphState.State>> => {
  console.log("---CHECKING DOC RELEVANCY---");

  const { messages } = state;
  const toolForScore = {
    name: "give_relevance_score",
    description: "Give a binary relevance score.",
    schema: z.object({
      binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
    }),
  };

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader for assessing the relevance of retrieved docs to a user query.
    Here are the retrieved docs:
    \n-------\n
    {docs}
    \n-------\n
    User query: {query}
    Using strict criteria, answer with a binary score ('yes' or 'no')`
  );

  const llmWithTools = llm.bindTools([toolForScore], {
    tool_choice: { type: "function", function: { name: toolForScore.name } },
  });

  const chain = prompt.pipe(llmWithTools);
  const lastMessage = messages[messages.length - 1];

  const score = await chain.invoke({
    query: messages[0].content as string,
    docs: lastMessage.content as string,
  });

  const binaryScore = score.tool_calls?.[0]?.args?.binaryScore;
  console.log("Doc relevance:", binaryScore);

  // Update relevant_documents if docs are relevant
  const relevant_documents: Document[] =
    binaryScore === "yes"
      ? [new Document({ pageContent: lastMessage.content as string })]
      : [];

  // Optionally add a follow-up message if not relevant
  if (binaryScore === "no") {
    const userQuery = messages[0].content as string;
    return {
      relevant_documents,
      messages: [
        ...state.messages,
        new AIMessage(
          `I couldn't find any specific information about "${userQuery}" in the retrieved documents.`
        ),
      ],
    };
  }

  return { relevant_documents };
};

// 4. generateRelevantResponse: Use fetched docs to generate a response
const generateRelevantResponse = async (
  state: typeof graphState.State
): Promise<Partial<typeof graphState.State>> => {
  console.log("---GENERATING RELEVANT RESPONSE---");

  const { messages } = state;
  const docs = state.relevant_documents as Document[];
  const docsContent = formatDocs(docs);

  const prompt = PromptTemplate.fromTemplate(
    `You are an assistant for query-answering tasks.
    Use the following retrieved context to answer the query.
    If the context does not include information about the query, respond with:
    "I don't have information about {query} in the provided documents."

    Context:
    {context}

    Query:
    {query}`
  );

  const chain = prompt.pipe(llm);
  const response = await chain.invoke({
    query: messages[0].content as string,
    context: docsContent,
  });

  return { messages: [response] };
};

// 5. checkHallucination: Ensure answer is grounded in the docs
const checkHallucination = async (
  state: typeof graphState.State
): Promise<Partial<typeof graphState.State>> => {
  console.log("---CHECKING HALLUCINATION---");

  const { relevant_documents, messages } = state;
  const toolForHallucination = {
    name: "checkHallucination",
    description:
      "Check if the response is grounded in the retrieved documents.",
    schema: z.object({
      binaryScore: z.string().describe("Answer grounded 'yes' or 'no'"),
      explanation: z.string().describe("Short explanation for the assessment"),
    }),
  };

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader.
    Retrieved Facts:
    {relevantDocs}

    Generated Answer:
    {generation}

    Check if the answer is strictly supported by the retrieved facts.
    Answer with a binary score ('yes' or 'no') and a brief explanation.`
  );

  const llmWithTools = llm.bindTools([toolForHallucination], {
    tool_choice: {
      type: "function",
      function: { name: toolForHallucination.name },
    },
  });

  const chain = prompt.pipe(llmWithTools);
  const lastMessage = messages[messages.length - 1];

  const score = await chain.invoke({
    generation: lastMessage.content as string,
    relevantDocs: formatDocs(relevant_documents),
  });

  if (score.tool_calls?.[0]?.args?.binaryScore === "no") {
    const originalQuery = messages[0].content as string;
    return {
      messages: [
        score,
        new AIMessage(
          `I don't have specific information about "${originalQuery}" in the provided documents.`
        ),
      ],
    };
  }

  return { messages: [score] };
};

// 6. generate: Final node—create the answer based on the user's query and the confirmed relevant docs.
const generate = async (
  state: typeof graphState.State
): Promise<Partial<typeof graphState.State>> => {
  console.log("---GENERATING FINAL RESPONSE---");
  const { messages } = state;
  const docs = state.relevant_documents as Document[];
  const docsContent = formatDocs(docs);

  const promptTemplate = ChatPromptTemplate.fromTemplate(`
    You are an assistant for query-answering tasks.
    Answer the query "{query}" using the context below.
    If no relevant context is present, respond with:
    "I don't have information about {query} in the provided documents."

    Context:
    {context}
  `);

  const response = await promptTemplate.pipe(llm).invoke({
    query: messages[0].content as string,
    context: docsContent,
  });

  console.log("User query:", messages[0].content as string);
  return { messages: [response] };
};

// -------------------------------------------------------------------------
// Graph Builder (Updated with Memory/Checkpoint support)
// -------------------------------------------------------------------------
const graphBuilder = new StateGraph(graphState)
  .addNode("retrieveOrRespond", retrieveOrRespond)
  .addNode("tools", tools)
  .addNode("checkDocRelevancy", checkDocRelevancy)
  .addNode("generateRelevantResponse", generateRelevantResponse)
  .addNode("checkHallucination", checkHallucination)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieveOrRespond")
  .addConditionalEdges("retrieveOrRespond", toolsCondition, {
    __end__: "__end__",
    tools: "tools",
  })
  .addEdge("tools", "checkDocRelevancy")
  .addEdge("checkDocRelevancy", "generateRelevantResponse")
  .addEdge("generateRelevantResponse", "checkHallucination")
  .addEdge("checkHallucination", "generate")
  .addEdge("generate", "__end__");

// Add checkpointer for state persistence
const checkpointer = new MemorySaver();
const graphWithMemory = graphBuilder.compile({ checkpointer });

// -------------------------------------------------------------------------
// Helper: Pretty Print Messages
// -------------------------------------------------------------------------
const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message.getType()}]: ${message.content}`;
  if (isAIMessage(message) && message.tool_calls?.length) {
    const toolCalls = message.tool_calls
      .map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += `\nTools:\n${toolCalls}`;
  }
  console.log(txt);
};

// let inputs1 = { messages: [new HumanMessage("Hello")] };

// for await (const step of await graphWithMemory.stream(inputs1, {
//   streamMode: "values",
// })) {
//   const lastMessage = step.messages[step.messages.length - 1];
//   prettyPrint(lastMessage);
//   console.log("-----\n");
// }

// -------------------------------------------------------------------------
// Demo: Memory Persistence Across Queries
// -------------------------------------------------------------------------
async function demonstrateMemoryPersistence() {
  const threadId = "demo-thr";
  const threadConfig = {
    configurable: { thread_id: threadId },
    streamMode: "values" as const,
  };

  console.log("FIRST QUERY:");
  const query1 = {
    messages: [new HumanMessage("What is polling")],
  };

  console.log("FIRST MESSAGE CONTENT:");
  for await (const step of await graphWithMemory.stream(query1, threadConfig)) {
    const lastMessage = step.messages[step.messages.length - 1];
    prettyPrint(lastMessage);
    console.log("-----\n");
  }

  console.log("SECOND QUERY:");
  const query2 = {
    messages: [new HumanMessage("List some ways of doing it")],
  };

  console.log("SECOND MESSAGE CONTENT:");
  for await (const step of await graphWithMemory.stream(query2, threadConfig)) {
    const lastMessage = step.messages[step.messages.length - 1];
    prettyPrint(lastMessage);
    console.log("-----\n");
  }
}

demonstrateMemoryPersistence();
