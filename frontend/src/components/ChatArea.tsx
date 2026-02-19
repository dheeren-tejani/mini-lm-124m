import { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { MessageCircle } from 'lucide-react';
import { ChatMessage } from '@/lib/types';
import { TypingIndicator } from './TypingIndicator';

interface ChatAreaProps {
  messages: ChatMessage[];
  isLoading: boolean;
}

function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4 text-center">
      <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mb-6">
        <MessageCircle className="w-8 h-8 text-muted-foreground" />
      </div>
      <h2 className="text-xl font-semibold text-foreground mb-2">Start a New Conversation</h2>
      <p className="text-sm text-muted-foreground max-w-md leading-relaxed">
        Experimental!! This model rambles a lot, it is not comparable to production ready models like Gemini, ChatGPT, etc, but it performs quite good in story telling/rambling
      </p>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  const isError = message.role === 'assistant' && message.content.startsWith('⚠️');

  return (
    <div className={`flex gap-2 animate-message-in ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-muted flex items-center justify-center text-xs font-bold text-foreground shrink-0 mt-0.5">
          N
        </div>
      )}
      <div
        className={`max-w-[75%] ${
          isUser
            ? 'bg-primary text-primary-foreground px-4 py-2.5 rounded-2xl rounded-br-md'
            : isError
            ? 'text-destructive text-sm'
            : 'text-foreground text-sm markdown-content'
        }`}
      >
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        ) : (
          <ReactMarkdown
            components={{
              code({ className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                const codeString = String(children).replace(/\n$/, '');
                if (match) {
                  return (
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={match[1]}
                      PreTag="div"
                      customStyle={{
                        background: 'hsl(0 0% 6.7%)',
                        border: '1px solid hsl(0 0% 12.2%)',
                        borderRadius: '0.5rem',
                        padding: '1rem',
                        fontSize: '0.825rem',
                        margin: '0.5rem 0',
                      }}
                    >
                      {codeString}
                    </SyntaxHighlighter>
                  );
                }
                return (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
}

export function ChatArea({ messages, isLoading }: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  if (messages.length === 0 && !isLoading) {
    return <EmptyState />;
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="max-w-2xl mx-auto space-y-4">
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isLoading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
