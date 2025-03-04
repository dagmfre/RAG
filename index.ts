import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey:
    "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",
});

async function getCompletion() {
  const completion = await client.chat.completions.create(
    {
      model: "deepseek/deepseek-r1:free",
      messages: [
        {
          role: "user",
          content: "What is the meaning of life?",
        },
      ],
    },
    {
      headers: {
        "HTTP-Referer": "<YOUR_SITE_URL>", // Optional. Site URL for rankings on openrouter.ai
        "X-Title": "<YOUR_SITE_NAME>", // Optional. Site title for rankings on openrouter.ai
      },
    }
  );
  console.log(completion.choices[0].message.content);
}

getCompletion();
