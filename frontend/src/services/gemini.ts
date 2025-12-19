import { GoogleGenerativeAI, GenerativeModel } from '@google/generative-ai';

class GeminiService {
  private genAI: GoogleGenerativeAI;
  private primaryModel: GenerativeModel;
  private secondaryModel: GenerativeModel;

  constructor() {
    const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('Gemini API key not found in environment variables');
    }

    this.genAI = new GoogleGenerativeAI(apiKey);
    // Use Gemini 2.5 Pro as primary; keep a lighter model as fallback
    const systemInstruction = 'You are a helpful coding assistant. Always leverage the prior messages as conversational context. When the user uses pronouns or says "both" or "the above", infer the referenced entities from the immediately preceding assistant/user turns.';
    this.primaryModel = this.genAI.getGenerativeModel({ model: 'gemini-2.5-pro', systemInstruction });
    this.secondaryModel = this.genAI.getGenerativeModel({ model: 'gemini-1.5-flash', systemInstruction });
  }

  async generateResponse(prompt: string): Promise<string> {
    const tryOnce = async (model: GenerativeModel): Promise<string> => {
      const result = await model.generateContent(prompt);
      return (await result.response).text();
    };

    try {
      return await this.withBackoff(() => tryOnce(this.primaryModel));
    } catch (primaryErr: any) {
      // If the primary model fails (quota/rate), try the secondary model
      if (this.isRateLimited(primaryErr)) {
        try {
          return await this.withBackoff(() => tryOnce(this.secondaryModel));
        } catch (secondaryErr: any) {
          if (this.isRateLimited(secondaryErr)) {
            return this.getMockResponse(prompt);
          }
          throw this.normalizeError(secondaryErr);
        }
      }
      throw this.normalizeError(primaryErr);
    }
  }

  async generateStreamResponse(prompt: string): Promise<ReadableStream<string>> {
    try {
      const result = await this.primaryModel.generateContentStream(prompt);
      return new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of result.stream) {
              const text = chunk.text();
              if (text) controller.enqueue(text);
            }
            controller.close();
          } catch (err: any) {
            if (err && (err.status === 429 || String(err.message || '').includes('quota'))) {
              // Fallback: non-streaming with secondary model; chunk locally
              try {
                const fallbackResult = await (await Promise.resolve(undefined), await (async () => {
                  return await (async () => {
                    return await (async () => this.secondaryModel.generateContent(prompt))();
                  })();
                })());
                const text = (await fallbackResult.response).text();
                const words = text.split(' ');
                for (let i = 0; i < words.length; i += 4) {
                  controller.enqueue(words.slice(i, i + 4).join(' ') + ' ');
                  await new Promise(r => setTimeout(r, 40));
                }
              } catch {
                controller.enqueue('The service is temporarily rate limited. Please try again soon.');
              }
              controller.close();
            } else {
              controller.error(err);
            }
          }
        }
      });
    } catch (err: any) {
      if (this.isRateLimited(err)) {
        const mock = this.getMockResponse(prompt);
        return new ReadableStream({
          start(controller) {
            controller.enqueue(mock);
            controller.close();
          }
        });
      }
      throw this.normalizeError(err);
    }
  }

  // Chat with history: returns a streaming response using prior turns for context
  async generateChatStream(history: Array<{ role: 'user' | 'assistant'; content: string }>): Promise<ReadableStream<string>> {
    if (!history.length) {
      return this.generateStreamResponse('');
    }

    // Split last user message from prior messages
    const last = history[history.length - 1];
    if (last.role !== 'user') {
      throw new Error('Last message must be from user to start a chat turn');
    }
    const prior = history.slice(0, -1).map((m) => ({
      role: m.role === 'assistant' ? 'model' as const : 'user' as const,
      parts: [{ text: m.content }]
    }));

    const sendWith = async (model: GenerativeModel): Promise<ReadableStream<string>> => {
      const chat = model.startChat({ history: prior });
      const res = await chat.sendMessageStream(last.content);
      return new ReadableStream<string>({
        async start(controller) {
          try {
            for await (const chunk of res.stream) {
              const t = chunk.text();
              if (t) controller.enqueue(t);
            }
            controller.close();
          } catch (e) {
            controller.error(e);
          }
        }
      });
    };

    try {
      return await this.withBackoff(() => sendWith(this.primaryModel));
    } catch (err: any) {
      if (this.isRateLimited(err)) {
        try {
          return await this.withBackoff(() => sendWith(this.secondaryModel));
        } catch (err2: any) {
          if (this.isRateLimited(err2)) {
            const mock = this.getMockResponse(last.content);
            return new ReadableStream<string>({
              start(controller) {
                controller.enqueue(mock);
                controller.close();
              }
            });
          }
          throw this.normalizeError(err2);
        }
      }
      throw this.normalizeError(err);
    }
  }

  private async withBackoff<T>(fn: () => Promise<T>, retries = 2): Promise<T> {
    let attempt = 0;
    let delayMs = 500;
    // eslint-disable-next-line no-constant-condition
    while (true) {
      try {
        return await fn();
      } catch (err: any) {
        if (attempt >= retries || !this.isRateLimited(err)) {
          throw err;
        }
        await new Promise((r) => setTimeout(r, delayMs));
        attempt += 1;
        delayMs *= 2;
      }
    }
  }

  private isRateLimited(error: unknown): boolean {
    const e = error as any;
    const msg = String(e?.message || '').toLowerCase();
    return e?.status === 429 || msg.includes('quota') || msg.includes('rate') || msg.includes('exceeded');
  }

  private normalizeError(error: unknown): Error {
    const e = error as any;
    const msg = e?.message || 'Unknown error calling Gemini API';
    return new Error(`Gemini API Error: ${msg}`);
  }

  private getMockResponse(message: string): string {
    const responses = [
      `I'm currently experiencing high demand and API limits. Here's a helpful response about "${message.slice(0, 80)}". Please try again in a little while.`,
      'The free-tier quota appears exhausted at the moment. Retrying in a minute often works.',
      'Temporary rate limit reached. I will be ready to answer shortly if you try again.'
    ];
    return responses[Math.floor(Math.random() * responses.length)];
  }
}

export const geminiService = new GeminiService();
export type { GenerativeModel } from '@google/generative-ai';
