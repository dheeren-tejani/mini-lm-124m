class BackendService {
  private baseUrl: string;

  constructor() {
    // Use environment variable or default to localhost:8000
    this.baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
  }

  /**
   * Format chat history into a prompt string for the backend
   */
  private formatChatHistory(history: Array<{ role: 'user' | 'assistant'; content: string }>): string {
    let prompt = '';
    for (const msg of history) {
      if (msg.role === 'user') {
        // Clean user message - ensure no trailing "AI:" or "User:" patterns
        const cleanedContent = msg.content.trim().replace(/\n*(User:|AI:)\s*/g, '').trim();
        prompt += `User: ${cleanedContent}\nAI:`;
      } else {
        // Clean assistant message - remove any "AI:" prefixes that might have been added
        const cleanedContent = msg.content.trim().replace(/^(AI:|User:)\s*/i, '').trim();
        // Only add content if it's not empty
        if (cleanedContent) {
          prompt += ` ${cleanedContent}\n`;
        }
      }
    }
    return prompt.trim();
  }

  /**
   * Generate a streaming response from the backend
   */
  async generateChatStream(
    history: Array<{ role: 'user' | 'assistant'; content: string }>
  ): Promise<ReadableStream<string>> {
    if (!history.length) {
      return new ReadableStream({
        start(controller) {
          controller.close();
        }
      });
    }

    const last = history[history.length - 1];
    if (last.role !== 'user') {
      throw new Error('Last message must be from user to start a chat turn');
    }

    // Format the full conversation history as a prompt
    const prompt = this.formatChatHistory(history);

    try {
      const response = await fetch(`${this.baseUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          max_tokens: 200,
          temperature: 0.8  // Slightly higher temperature for more variety and less repetition
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend API error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body reader available');
      }

      const decoder = new TextDecoder();

      return new ReadableStream<string>({
        async start(controller) {
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              const chunk = decoder.decode(value, { stream: true });
              if (chunk) {
                controller.enqueue(chunk);
              }
            }
            controller.close();
          } catch (error) {
            controller.error(error);
          }
        }
      });
    } catch (error) {
      throw new Error(`Failed to connect to backend: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Generate a simple streaming response (for title generation, etc.)
   */
  async generateStreamResponse(prompt: string): Promise<ReadableStream<string>> {
    try {
      const response = await fetch(`${this.baseUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          max_tokens: 50,
          temperature: 0.7
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend API error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body reader available');
      }

      const decoder = new TextDecoder();

      return new ReadableStream<string>({
        async start(controller) {
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              const chunk = decoder.decode(value, { stream: true });
              if (chunk) {
                controller.enqueue(chunk);
              }
            }
            controller.close();
          } catch (error) {
            controller.error(error);
          }
        }
      });
    } catch (error) {
      throw new Error(`Failed to connect to backend: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}

export const backendService = new BackendService();

