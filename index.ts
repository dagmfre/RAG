import "puppeteer";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import {
  MemorySaver,
  MessagesAnnotation,
  StateGraph,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import dotenv from "dotenv";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
dotenv.config();
import { ChatMistralAI } from "@langchain/mistralai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  isAIMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { v4 as uuidv4 } from "uuid";

// Load and chunk contents of blog
const webLoader = new PuppeteerWebBaseLoader(
  "https://addissoftware.com/making-the-leap-why-switching-to-rtk-query-from-saga-makes-sense",
  {
    launchOptions: {
      headless: true,
    },
    gotoOptions: {
      waitUntil: "networkidle0", // Wait until network is completely idle
      timeout: 60000, // Increase timeout to 60 seconds
    },
    // This is where the magic happens - custom evaluation function
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
  model: "mistral-small-latest", // or other Llama models
  temperature: 0,
  maxRetries: 2,
});

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACEHUB_API_KEY,
  model: "BAAI/bge-m3",
});

const vectorStore = await MemoryVectorStore.fromDocuments(
  allSplits,
  embeddings
);

// Index chunks
await vectorStore.addDocuments(allSplits);

////////////////////////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////////////////////////
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

  const docsContent = toolMessages.map((doc) => doc.content).join("\n");

  // creating systemMessage from the retrieved docs
  const systemMessageContent =
    "You are an assistant for question-answering tasks. " +
    "Use the following pieces of retrieved context to answer " +
    "the question. If you don't know the answer, say that you " +
    "don't know. Use three sentences maximum and keep the " +
    "answer concise." +
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
  return { messages: [response] };
}

const graphBuilder = new StateGraph(MessagesAnnotation)
  .addNode("retrieveOrRespond", retrieveOrRespond)
  .addNode("tools", tools)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieveOrRespond")
  .addConditionalEdges("retrieveOrRespond", toolsCondition, {
    __end__: "__end__",
    tools: "tools",
  })
  .addEdge("tools", "generate")
  .addEdge("generate", "__end__");

/////////////////////////////////////////////////////////////////////////////////////////////
// Add checkpointer using MemorySaver for chat history persistence
const checkpointer = new MemorySaver();
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

  // // First query
  console.log("FIRST QUERY:");
  const query1 = {
    messages: [
      {
        role: "user",
        content: "My name is dagm",
      },
    ],
  };

  console.log("-----\n");

  // Stream the response for the first query
  for await (const step of await graphWithMemory.stream(query1, threadConfig)) {
    const lastMessage = step.messages[step.messages.length - 1];
    prettyPrint(lastMessage);
  }

  // Second query - demonstrating memory persistence
  // console.log("\nSECOND QUERY (with memory of first conversation):");
  // const query2 = {
  //   messages: [
  //     {
  //       role: "user",
  //       content: "What is my name?",
  //     },
  //   ],
  // };

  // // Stream the response for the second query
  // for await (const step of await graphWithMemory.stream(query2, threadConfig)) {
  //   const lastMessage = step.messages[step.messages.length - 1];
  //   prettyPrint(lastMessage);
  //   console.log("-----\n");
  // }
}

demonstrateMemoryPersistence();
