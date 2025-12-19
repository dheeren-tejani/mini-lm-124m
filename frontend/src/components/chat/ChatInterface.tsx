import { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInputResponsive';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { backendService } from '@/services/backend';
import { createChat, getMessages as loadMessages, setMessages as saveMessages, setChatTitle, getChats } from '@/lib/chatStore';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  attachments?: File[];
}

export const ChatInterface = () => {
  const { chatId } = useParams<{ chatId: string }>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chatId) return;
    // Ensure chat exists in store (handles deep link to /chat/:id)
    const exists = getChats().some(c => c.id === chatId);
    if (!exists) createChat(chatId);
    const stored = loadMessages(chatId).map(m => ({ ...m, timestamp: new Date(m.timestamp) })) as unknown as Message[];
    setMessages(stored);
  }, [chatId]);

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (content: string, attachments?: File[]) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
      attachments,
    };

    let currentMessages: Message[] = [];
    setMessages(prev => {
      const next = [...prev, userMessage];
      currentMessages = next; // Capture the updated messages
      if (chatId) {
        saveMessages(chatId, next.map(m => ({ ...m, timestamp: m.timestamp.toISOString() })) as any);
      }
      return next;
    });
    setIsLoading(true);

    try {
      // Ensure we have the latest messages (fallback to loading from store if needed)
      if (chatId && currentMessages.length === 1) {
        // If this is the first message, make sure we loaded any existing messages
        const stored = loadMessages(chatId).map(m => ({ ...m, timestamp: new Date(m.timestamp) })) as unknown as Message[];
        if (stored.length > currentMessages.length) {
          currentMessages = [...stored, userMessage];
        }
      }
      
      // Build history using the current messages (includes all previous messages and the new user message)
      const history = currentMessages.map(m => ({ role: m.role, content: m.content }));
      const stream = await backendService.generateChatStream(history);

      // Create a placeholder assistant message and stream into it
      const id = (Date.now() + 1).toString();
      const assistantMessage: Message = {
        id,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
      };
      setMessages(prev => {
        const next = [...prev, assistantMessage];
        if (chatId) {
          saveMessages(chatId, next.map(m => ({ ...m, timestamp: m.timestamp.toISOString() })) as any);
        }
        return next;
      });

      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let done = false;
      while (!done) {
        const { value, done: isDone } = await reader.read();
        done = isDone;
        if (value) {
          const chunk = typeof value === 'string' ? value : decoder.decode(value);
          setMessages(prev => {
            const next = prev.map(m => m.id === id ? { ...m, content: m.content + chunk } : m);
            if (chatId) {
              saveMessages(chatId, next.map(m => ({ ...m, timestamp: m.timestamp.toISOString() })) as any);
            }
            return next;
          });
        }
      }

      // Generate a short title from the first user message and stream it to sidebar
      if (chatId && currentMessages.length === 1) {
        try {
          const titleStream = await backendService.generateStreamResponse(
            `Generate a concise 3-6 word chat title for: ${content}. No quotes.`
          );
          const titleReader = titleStream.getReader();
          const dec = new TextDecoder();
          let titleDone = false;
          let title = '';
          while (!titleDone) {
            const { value: tval, done: tdone } = await titleReader.read();
            titleDone = tdone;
            if (tval) {
              title += typeof tval === 'string' ? tval : dec.decode(tval);
              setChatTitle(chatId, title);
            }
          }
        } catch { }
      }
    } catch (error) {
      console.error('Error getting AI response:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockResponse = (userInput: string): string => {
    const responses = [
      `I understand you're asking about "${userInput}". Here's my response:

## Analysis

This is an interesting question that requires careful consideration.

### Key Points

- **Point 1**: Detailed explanation here
- **Point 2**: Another important aspect  
- **Point 3**: Additional considerations

\`\`\`javascript
// Example code snippet
function exampleFunction() {
  console.log("This is a code example");
  return "response";
}
\`\`\`

### Mathematical Example

For instance, the quadratic formula: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$

Or a more complex equation:
$$\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}$$

Would you like me to elaborate on any of these points?`,

      `Thank you for your question about "${userInput}".

| Aspect | Description | Importance |
|--------|-------------|------------|
| Technical | Implementation details | High |
| Practical | Real-world usage | Medium |
| Theoretical | Conceptual understanding | High |

## Mathematical Concepts

Here's a probability density function:
$$f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2}$$

> **Important**: This is a comprehensive response that covers multiple aspects of your question.

Let me know if you need more specific information!`,
    ];

    return responses[Math.floor(Math.random() * responses.length)];
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Messages Area */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="max-w-none sm:max-w-4xl mx-auto px-3 sm:px-6 lg:px-8">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center min-h-[50vh] sm:min-h-[60vh] px-6 sm:px-8">
              <div className="text-center max-w-xs sm:max-w-2xl animate-fade-in-up space-y-6 sm:space-y-8">
                {/* Welcome Icon */}
                <div className="relative">
                  <div className="w-16 h-16 sm:w-20 sm:h-20 mx-auto bg-gradient-to-br from-primary/15 to-accent/15 rounded-3xl flex items-center justify-center shadow-xl animate-bounce-in">
                    <span className="text-2xl sm:text-3xl">💬</span>
                  </div>
                  <div className="absolute -inset-4 sm:-inset-6 bg-gradient-to-r from-primary/3 via-transparent to-accent/3 rounded-full blur-2xl animate-pulse-glow" />
                </div>

                {/* Welcome Text */}
                <div className="space-y-3 sm:space-y-4">
                  <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold bg-gradient-to-r from-foreground via-foreground/90 to-foreground/70 bg-clip-text text-transparent">
                    Start a New Conversation
                  </h2>
                  <p className="text-sm sm:text-base lg:text-lg text-muted-foreground/80 leading-relaxed">
                    Ask me anything and I'll help you with detailed, formatted responses including
                    <span className="text-primary font-medium"> code</span>,
                    <span className="text-accent font-medium"> tables</span>, and
                    <span className="text-secondary-foreground font-medium"> math equations</span>.
                  </p>
                </div>

                {/* Feature Pills */}
                <div className="flex flex-wrap justify-center gap-2 sm:gap-3">
                  {[
                    { icon: "💻", label: "Code Highlighting" },
                    { icon: "📊", label: "Tables & Charts" },
                    { icon: "🔢", label: "LaTeX Math" },
                    { icon: "📎", label: "File Uploads" }
                  ].map((feature, index) => (
                    <div
                      key={feature.label}
                      className={cn(
                        "flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-1.5 sm:py-2 rounded-full",
                        "bg-gradient-to-r from-accent/8 to-accent/4",
                        "text-accent-foreground shadow-sm",
                        "hover:from-accent/15 hover:to-accent/8 transition-all duration-300",
                        "hover:scale-105 cursor-default animate-fade-in"
                      )}
                      style={{ animationDelay: `${0.5 + index * 0.1}s` }}
                    >
                      <span className="text-sm sm:text-base">{feature.icon}</span>
                      <span className="text-xs sm:text-sm font-medium">{feature.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
            </div>
          )}

          {isLoading && (
            <div className="flex gap-6 p-6 bg-gradient-to-r from-chat-bubble-assistant/30 to-transparent animate-fade-in rounded-2xl mx-4">
              <div className="flex-shrink-0 animate-scale-in">
                <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-secondary/15 to-accent/8 flex items-center justify-center shadow-lg">
                  <div className="w-5 h-5 border-2 border-secondary-foreground/60 border-t-secondary-foreground rounded-full animate-spin" />
                </div>
              </div>
              <div className="flex-1 min-w-0 space-y-2">
                <div className="text-sm font-semibold text-foreground">Assistant</div>
                <div className="flex items-center gap-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <span className="text-muted-foreground animate-pulse">Thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {/* Input Area */}
      <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
    </div>
  );
};