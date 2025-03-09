import "puppeteer";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { PromptTemplate } from "@langchain/core/prompts";
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

// New tool: Document Relevancy Checker
const checkDocRelevancySchema = z.object({
  query: z.string(),
  document: z.string(),
});

const checkDocRelevancy = tool(
  async ({ query, document }) => {
    const systemMessage =
      "You are a grader assessing relevance of a retrieved document to a user question. " +
      "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. " +
      "It does not need to be a stringent test. The goal is to filter out erroneous retrievals. " +
      "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.";

    const messages = [
      new SystemMessage(systemMessage),
      new HumanMessage(
        `Retrieved document: \n\n ${document} \n\n User question: ${query}`
      ),
    ];

    const response = await llm.invoke(messages);

    const isRelevant =
      typeof response.content === "string" &&
      response.content.toLowerCase().includes("yes");

    return isRelevant ? "yes" : "no";
  },
  {
    name: "checkDocRelevancy",
    description: "Check if a document is relevant to a query.",
    schema: checkDocRelevancySchema,
  }
);

// New tool: Hallucination Checker
const checkHallucinationSchema = z.object({
  documents: z.string(),
  generation: z.string(),
});

const checkHallucination = tool(
  async ({ documents, generation }) => {
    const systemMessage =
      "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. " +
      "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.";

    const messages = [
      new SystemMessage(systemMessage),
      new HumanMessage(
        `Set of facts: \n\n ${documents} \n\n LLM generation: ${generation}`
      ),
    ];

    const response = await llm.invoke(messages);

    // Parse the response to get yes/no
    const isGrounded =
      typeof response.content === "string" &&
      response.content.toLowerCase().includes("yes");

    return isGrounded ? "yes" : "no";
  },
  {
    name: "checkHallucination",
    description:
      "Check if a generated response is grounded in the retrieved documents.",
    schema: checkHallucinationSchema,
  }
);

// New tool: Document Highlighter
const highlightDocsSchema = z.object({
  documents: z.string(),
  question: z.string(),
  generation: z.string(),
});

const highlightDocs = tool(
  async ({ documents, question, generation }) => {
    const systemMessage =
      "You are an advanced assistant for document search and retrieval. You are provided with:\n" +
      "1. A question.\n" +
      "2. A generated answer based on the question.\n" +
      "3. A set of documents that were referenced in generating the answer.\n\n" +
      "Your task is to identify and extract the exact inline segments from the provided documents " +
      "that directly correspond to the content used to generate the given answer. The extracted " +
      "segments must be verbatim snippets from the documents, ensuring a word-for-word match with " +
      "the text in the provided documents.\n\n" +
      "Format your response as a JSON array with objects containing 'source' and 'segment' fields.";

    const messages = [
      new SystemMessage(systemMessage),
      new HumanMessage(
        `Documents: \n\n${documents}\n\n` +
          `Question: ${question}\n\n` +
          `Generated answer: ${generation}`
      ),
    ];

    const response = await llm.invoke(messages);
    return response.content;
  },
  {
    name: "highlightDocs",
    description:
      "Highlight segments from the documents that were used in the generated response.",
    schema: highlightDocsSchema,
  }
);

// A tool to make LLM include a tool-call(if retrieval is necessary) or directly respond
// then return/add to the message state: AIMessage
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

// A prebuilt langgragh node that runs created tools, here only the retrieval node
const tools = new ToolNode([
  retrieve,
  checkDocRelevancy,
  checkHallucination,
  highlightDocs,
]);

// Filter out irrelevant documents
const filterDocsNode = async (state: typeof MessagesAnnotation.State) => {
  // Get the last human message for the query
  const humanMessages = state.messages.filter(
    (message) => message instanceof HumanMessage
  );
  const lastHumanMessage = humanMessages[humanMessages.length - 1];
  const query = lastHumanMessage.content as string;

  // Get tool messages with retrieved docs
  let retrievedDocsToolMessage: ToolMessage | null = null;
  for (let i = state.messages.length - 1; i >= 0; i--) {
    if (
      state.messages[i] instanceof ToolMessage &&
      state.messages[i].name === "retrieve"
    ) {
      retrievedDocsToolMessage = state.messages[i] as ToolMessage;
      break;
    }
  }

  if (!retrievedDocsToolMessage) {
    return state;
  }

  // Extract the docs from the tool message
  const retrievedDocs = retrievedDocsToolMessage.artifact;

  // Filter relevant documents
  let relevantDocs = [];
  for (const doc of retrievedDocs) {
    const isRelevant = await llm.invoke([
      new SystemMessage(
        "You are a grader assessing relevance of a retrieved document to a user question. " +
          "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. " +
          "It does not need to be a stringent test. The goal is to filter out erroneous retrievals. " +
          "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
      ),
      new HumanMessage(
        `Retrieved document: \n\n ${doc.pageContent} \n\n User question: ${query}`
      ),
    ]);

    if (
      typeof isRelevant.content === "string" &&
      isRelevant.content.toLowerCase().includes("yes")
    ) {
      relevantDocs.push(doc);
    }
  }

  // Create a new tool message with filtered docs
  const serialized = relevantDocs
    .map((doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`)
    .join("\n");

  const newToolMessage = new ToolMessage(serialized);
  newToolMessage.name = "retrieve";
  newToolMessage.tool_name = "retrieve";
  newToolMessage.tool_call_id = retrievedDocsToolMessage.tool_call_id;
  newToolMessage.artifact = relevantDocs;

  // Replace the old tool message with the new one
  const newMessages = [...state.messages];
  for (let i = 0; i < newMessages.length; i++) {
    if (newMessages[i] === retrievedDocsToolMessage) {
      newMessages[i] = newToolMessage;
      break;
    }
  }

  return { messages: newMessages };
};

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
