import { r2rClient } from "r2r-js";
import { config } from "dotenv";

config();

const client = new r2rClient(process.env.R2R_API_URL as string); // Set API URL
client.setApiKey(process.env.R2R_API_KEY as string); // Set API key

const main = async (): Promise<void> => {
  // 1) Initiate a streaming RAG request
  const resultStream = await client.retrieval.rag({
    query: "Ways of customizing RAG in R2R",
    ragGenerationConfig: {
      model: "gpt-4.5-preview-2025-02-27",
      temperature: 0.2,
      // stream: true,
    },
    searchSettings: {
      filters: {
        document_id: "2b8d42dc-7c6c-52b7-9bc0-fabea114312f",
      },
    },
  });

  console.log(resultStream);
  

  // // 2) Check if we got an async iterator (streaming)
  // if (Symbol.asyncIterator in resultStream) {
  //   // 2a) Loop over each event from the server
  //   for await (const event of resultStream) {
  //     // First check if we're dealing with binary data (Uint8Array)
  //     if (event instanceof Uint8Array) {
  //       // Decode the binary data to a string
  //       const textDecoder = new TextDecoder();
  //       const text = textDecoder.decode(event);
  //       console.log("Decoded binary chunk:", text);
  //     } else {
  //       // Handle regular event objects
  //       switch (event.event) {
  //         case "search_results":
  //           console.log("Search results:", event.data);
  //           break;
  //         case "message":
  //           console.log("Partial message delta:", event.data.delta);
  //           break;
  //         case "citation":
  //           console.log("New citation event:", event.data);
  //           break;
  //         case "final_answer":
  //           console.log("Final answer:", event.data.generated_answer);
  //           break;
  //         // ... add more cases if you have other event types, e.g. tool_call / tool_result
  //         default:
  //           console.log("Unknown or unhandled event:", event);
  //       }
  //     }
  //   }
  // } else {
  //   // 2b) If streaming was NOT enabled or server didn't send SSE,
  //   //     we'd get a single response object instead.
  //   console.log("Non-streaming RAG response:", resultStream);
  // }
};

main();
