async function askQuestion(question: string) {
  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": "Bearer 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", // Replace with your actual API key
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        "model": "deepseek/deepseek-r1:free",
        "messages": [
          {
            "role": "user",
            "content": question
          }
        ]
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed with status: ${response.status}`);
    }

    const data = await response.json();
    return data.choices[0].message.content;
  } catch (error) {
    console.error("Error calling OpenRouter API:", error);
    throw error;
  }
}

// Example usage
askQuestion("What is the meaning of life?")
  .then(answer => console.log("Answer:", answer))
  .catch(error => console.error("Failed to get answer:", error));