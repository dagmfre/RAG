import { HfInference } from '@huggingface/inference';
import dotenv from "dotenv";
dotenv.config();

export class DirectHFClient {
  private client: HfInference;
  private model: string;

  constructor(apiKey: string, model: string = "meta-llama/Meta-Llama-3-8B-Instruct") {
    this.client = new HfInference(apiKey);
    this.model = model;
  }

  async invoke(messages: any[], tools?: any[]) {
    let payload: any = {
      model: this.model,
      messages: messages.map(msg => {
        if (msg.role === 'system' || msg.role === 'user' || msg.role === 'assistant') {
          return {
            role: msg.role,
            content: msg.content
          };
        } else if (msg.role === 'tool') {
          return {
            role: 'tool',
            content: msg.content,
            tool_call_id: msg.tool_call_id
          };
        }
        return msg;
      }),
      temperature: 0.1,
      max_tokens: 1024
    };

    // Add tools if provided
    if (tools && tools.length > 0) {
      payload.tools = tools;
    }

    try {
      const response = await this.client.textGeneration({
        inputs: JSON.stringify(payload),
        model: this.model,
        parameters: {
          return_full_text: false
        }
      });

      // Parse the response
      try {
        const parsedResponse = JSON.parse(response.generated_text);
        return parsedResponse;
      } catch (e) {
        // If not JSON, return as is
        return {
          role: 'assistant',
          content: response.generated_text
        };
      }
    } catch (error) {
      console.error("Error calling HuggingFace API:", error);
      throw error;
    }
  }
}