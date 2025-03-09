import "puppeteer";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import dotenv from "dotenv";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
dotenv.config();
import { ChatMistralAI } from "@langchain/mistralai";
import { MongoClient } from "mongodb";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import {
  BaseCheckpointSaver,
  MessagesAnnotation,
  StateGraph,
} from "@langchain/langgraph";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  isAIMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { v4 as uuidv4 } from "uuid";
import { StrOutputParser } from "langchain_core/output_parsers";

// MongoDB connection
const client = new MongoClient(`${process.env.MONGODB_URI}`);
await client.connect();
const db = client.db("sophyy");
const conversationsCollection = db.collection("conversations");
const indexedDocsCollection = db.collection("indexedDocs");

// Load and chunk contents of blog
const webLoader = new PuppeteerWebBaseLoader(
  "https://addissoftware.com/making-the-leap-why-switching-to-rtk-query-from-saga-makes-sense",
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

// Splitting
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1500,
  chunkOverlap: 300,
});
const allSplits = await splitter.splitDocuments(docs);

const llm = new ChatMistralAI({
  model: "mistral-small-latest",
  temperature: 0,
  maxRetries: 2,
});

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACEHUB_API_KEY,
  model: "BAAI/bge-m3",
});

// Pinecone setup
const pinecone = new PineconeClient();

const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX!);

const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex,
  // Maximum number of batch requests to allow at once. Each batch is 1000 vectors.
  maxConcurrency: 5,
  namespace: "AddisBlog",
});

// Index chunks
await vectorStore.addDocuments(allSplits);

// Create a retrieval Tool
const retrieveSchema = z.object({ query: z.string() });

const retrieve = tool(
  async ({ query }) => {
    const retrievedDocs = await vectorStore.similaritySearch(query, 5);
    const serialized = retrievedDocs
      .map(
        (doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`
      )
      .join("\n");
    return [serialized, retrievedDocs];
  },
  {
    name: "retrieve",
    description: "Retrieve information related to a query.",
    schema: retrieveSchema,
    responseFormat: "content_and_artifact",
  }
);

const retrieveOrRespond = async (state: typeof MessagesAnnotation.State) => {
  const llmWithTools = llm.bindTools([retrieve]);

  // Extract human messages from the state
  const messages = state.messages.filter(
    (message) =>
      message instanceof HumanMessage || message instanceof SystemMessage
  );

  // Invoke the LLM with the messages from state
  const response = await llmWithTools.invoke(messages);
  return { messages: [response] };
};

const checkDocRelevancy = async (
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> => {
  console.log("---GET RELEVANCE---");

  const { messages } = state;

  const tool = {
    name: "give_relevance_score",
    description: "Give a relevance score to the retrieved documents.",
    schema: z.object({
      binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
    }),
  };

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing relevance of retrieved docs to a user question.
  Here are the retrieved docs:
  \n ------- \n
  {context} 
  \n ------- \n
  Here is the user question: {question}
  If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
  It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
  Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
  Yes: The docs are relevant to the question.
  No: The docs are not relevant to the question.`
  );

  const llmWithTools = llm.bindTools([tool], {
    tool_choice: { type: "function", function: { name: tool.name } },
  });

  const chain = prompt.pipe(llmWithTools);

  const lastMessage = messages[messages.length - 1];

  console.log(messages);

  const score = await chain.invoke({
    question: messages[0].content as string,
    context: lastMessage.content as string,
  });

  return {
    messages: [score],
  };
};

// New tool: Hallucination Checker
const checkHallucinationSchema = z.object({
  documents: z.string(),
  generation: z.string(),
});

const checkHallucination = async (
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> => {
  console.log("---GET RELEVANCE---");

  const { messages } = state;

  const tool = {
    name: "give_relevance_score",
    description: "Give a relevance score to the retrieved documents.",
    schema: z.object({
      binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
    }),
  };

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Here are the set of retrieved facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the LLM generation: {generation}
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.`
  );

  const llmWithTools = llm.bindTools([tool], {
    tool_choice: { type: "function", function: { name: tool.name } },
  });

  const chain = prompt.pipe(llmWithTools);

  const lastMessage = messages[messages.length - 1];

  console.log(messages);

  const score = await chain.invoke({
    generation: lastMessage.content as string,
    documents: messages[0].content as string,
  });

  return {
    messages: [score],
  };
};

// A tool to make LLM include a tool-call(if retrieval is necessary) or directly respond
// then return/add to the message state: AIMessage

// A prebuilt langgragh node that runs created tools, here only the retrieval node
const tools = new ToolNode([retrieve]);

// Generate final response from the llm by:
// - first getting the retrieved docs inside toolMessages: array of tool messages(ToolMessage[]).
// - using those retrieved docs to create a systemMessageContent
// - creating a clean and filtered dialogue message: conversationMessages
// - create final prompt from the systemMessageContent + conversationMessages
async function generate(state: typeof MessagesAnnotation.State) {
  // Get generated ToolMessages
  let recentToolMessages: ToolMessage[] = [];

  const messages = state.messages as Array<
    AIMessage | HumanMessage | ToolMessage
  >;
  for (let i = messages.length - 1; i >= 0; i--) {
    let message = state["messages"][i];
    if (message instanceof ToolMessage) {
      recentToolMessages.push(message);
    } else {
      break;
    }
  }
  let toolMessages: ToolMessage[] = recentToolMessages.reverse();

  const docsContent = toolMessages
    .filter((msg) => msg.name === "retrieve")
    .map((doc) => doc.content)
    .join("\n");

  // creating systemMessage from the retrieved docs
  const systemMessageContent =
    "You are an assistant for question-answering tasks. " +
    "Use the following pieces of retrieved context to answer " +
    "the question. If you don't know the answer, say that you " +
    "don't know. Use three sentences maximum and keep the " +
    "answer concise. Only include information that is directly supported by the provided context." +
    "\n\n" +
    `${docsContent}`;

  const conversationMessages = (
    state.messages as Array<AIMessage | HumanMessage>
  ).filter(
    (message) =>
      message instanceof HumanMessage ||
      message instanceof SystemMessage ||
      (message instanceof AIMessage && message.tool_calls?.length === 0)
  );

  // final prompt
  const prompt = [
    new SystemMessage(systemMessageContent),
    ...conversationMessages,
  ];

  // Run
  const response = await llm.invoke(prompt);

  // Get the last human message
  const lastHumanMessage = conversationMessages
    .filter((msg) => msg instanceof HumanMessage)
    .pop();

  // Check for hallucinations
  if (docsContent && lastHumanMessage) {
    const isGrounded = await llm.invoke([
      new SystemMessage(
        "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. " +
          "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."
      ),
      new HumanMessage(
        `Set of facts: \n\n ${docsContent} \n\n LLM generation: ${response.content}`
      ),
    ]);

    // If hallucination detected, add a warning
    if (!isGrounded.content.toLowerCase().includes("yes")) {
      const warningContent = `${response.content}\n\n[Note: This response may contain information not fully supported by the retrieved documents.]`;
      return { messages: [new AIMessage(warningContent)] };
    }

    // Find document segments that support the answer
    const highlighting = await llm.invoke([
      new SystemMessage(
        "You are an advanced assistant for document search and retrieval. " +
          "Identify the exact segments from the provided documents that were used to generate the answer. " +
          "Format your response as: 'HIGHLIGHTED SEGMENTS:\n[segment 1]\n[segment 2]'"
      ),
      new HumanMessage(
        `Documents: \n\n${docsContent}\n\n` +
          `Question: ${lastHumanMessage.content}\n\n` +
          `Generated answer: ${response.content}`
      ),
    ]);

    // Add the highlighted segments as footnotes
    if (highlighting.content.includes("HIGHLIGHTED SEGMENTS:")) {
      const augmentedResponse = `${
        response.content
      }\n\n**Sources:**\n${highlighting.content
        .split("HIGHLIGHTED SEGMENTS:")[1]
        .trim()}`;
      return { messages: [new AIMessage(augmentedResponse)] };
    }
  }

  return { messages: [response] };
}

const graphBuilder = new StateGraph(MessagesAnnotation)
  .addNode("retrieveOrRespond", retrieveOrRespond)
  .addNode("tools", tools)
  .addNode("filterDocs", filterDocsNode)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieveOrRespond")
  .addConditionalEdges("retrieveOrRespond", toolsCondition, {
    __end__: "__end__",
    tools: "tools",
  })
  .addEdge("tools", "filterDocs")
  .addEdge("filterDocs", "generate")
  .addEdge("generate", "__end__");

// Add checkpointer using MongoDB for chat history persistence
const saveState = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const threadId = state.thread_id || uuidv4();
  await conversationsCollection.updateOne(
    { threadId },
    { $set: { messages } },
    { upsert: true }
  );
  return threadId;
};

const loadState = async (threadId) => {
  const conversation = await conversationsCollection.findOne({ threadId });
  if (conversation) {
    return { messages: conversation.messages, threadId };
  }
  return null;
};

const checkpointer: BaseCheckpointSaver = {
  save: saveState,
  load: loadState,
};

const graphWithMemory = graphBuilder.compile({ checkpointer });

// Helper function to print messages nicely
const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message._getType()}]: ${message.content}`;
  if ((isAIMessage(message) && message.tool_calls?.length) || 0 > 0) {
    const tool_calls = (message as AIMessage)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};

// Demo function to show persistence across multiple queries with the same thread ID
async function demonstrateMemoryPersistence() {
  // Create a thread ID
  const threadId = "demo-thread-id";

  // Configuration for streaming with thread ID
  const threadConfig = {
    configurable: { thread_id: threadId },
    streamMode: "values" as const,
  };

  // First query
  console.log("FIRST QUERY:");
  const query1 = {
    messages: [
      new HumanMessage(
        "Tell me about the benefits of RTK Query over Redux Saga"
      ),
    ],
  };

  console.log("-----\n");

  // Stream the response for the first query
  for await (const step of await graphWithMemory.stream(query1, threadConfig)) {
    const lastMessage = step.messages[step.messages.length - 1];
    prettyPrint(lastMessage);
  }

  // Second query - demonstrating memory persistence
  console.log("\nSECOND QUERY (with memory of first conversation):");
  const query2 = {
    messages: [new HumanMessage("What are the main differences between them?")],
  };

  // Stream the response for the second query
  for await (const step of await graphWithMemory.stream(query2, threadConfig)) {
    const lastMessage = step.messages[step.messages.length - 1];
    prettyPrint(lastMessage);
    console.log("-----\n");
  }
}

demonstrateMemoryPersistence();
