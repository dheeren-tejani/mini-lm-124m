import { useState } from 'react';
import { Copy, Check, User, Bot } from 'lucide-react';
import { MarkdownRenderer } from './MarkdownRenderer';
import { Button } from '../ui/enhanced-button';
import { cn } from '@/lib/utils';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  attachments?: File[];
}

interface ChatMessageProps {
  message: Message;
}

export const ChatMessageEnhanced = ({ message }: ChatMessageProps) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';
  
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  };

  const formatTime = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      weekday: 'short',
      month: 'short',
      day: 'numeric'
    }).format(date);
  };

  return (
    <div className={cn(
      "group relative animate-in slide-in-from-bottom-1 duration-500",
      "hover:bg-gradient-to-r hover:from-transparent hover:to-accent/[0.02]",
      "transition-all duration-300 rounded-2xl -mx-2 px-2 py-3"
    )}>
      <div className="flex gap-4 sm:gap-6">
        {/* Avatar */}
        <div className={cn(
          "flex-shrink-0 relative",
          "w-8 h-8 sm:w-10 sm:h-10 rounded-xl",
          "flex items-center justify-center shadow-sm",
          "transition-all duration-200 group-hover:shadow-md",
          isUser 
            ? "bg-gradient-to-br from-primary/10 to-primary/5 border border-primary/20" 
            : "bg-gradient-to-br from-accent/20 to-accent/10 border border-accent/30"
        )}>
          {isUser ? (
            <User className="w-4 h-4 sm:w-5 sm:h-5 text-primary/80" />
          ) : (
            <Bot className="w-4 h-4 sm:w-5 sm:h-5 text-accent-foreground/80" />
          )}
          
          {/* Status indicator for assistant */}
          {!isUser && (
            <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-green-400 border border-background shadow-sm" />
          )}
        </div>

        {/* Message Content */}
        <div className="flex-1 min-w-0 space-y-2">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-sm font-semibold text-foreground">
                {isUser ? 'You' : 'Noir Whisper'}
              </span>
              <time className="text-xs text-foreground-subtle font-mono">
                {formatTime(message.timestamp)}
              </time>
            </div>
            
            {/* Copy Button */}
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={handleCopy}
              className={cn(
                "opacity-0 group-hover:opacity-100 transition-all duration-200",
                "hover:bg-accent/20 hover:scale-110 focus-visible:opacity-100",
                "text-foreground-subtle hover:text-foreground"
              )}
            >
              {copied ? (
                <Check className="w-3 h-3 text-green-500" />
              ) : (
                <Copy className="w-3 h-3" />
              )}
            </Button>
          </div>

          {/* Attachments */}
          {message.attachments && message.attachments.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3">
              {message.attachments.map((file, index) => (
                <div
                  key={index}
                  className={cn(
                    "inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm",
                    "bg-gradient-to-r from-accent/20 to-accent/10",
                    "border border-accent/30 text-accent-foreground"
                  )}
                >
                  <span>📎</span>
                  <span className="font-medium truncate max-w-32">
                    {file.name}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Message Content */}
          <div className={cn(
            "prose prose-sm max-w-none",
            "prose-headings:text-foreground prose-headings:font-semibold",
            "prose-p:text-foreground-muted prose-p:leading-relaxed",
            "prose-strong:text-foreground prose-strong:font-semibold",
            "prose-code:text-foreground prose-code:bg-code-background",
            "prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md",
            "prose-pre:bg-code-background prose-pre:border prose-pre:border-code-border",
            "prose-blockquote:text-foreground-muted prose-blockquote:border-l-accent",
            "prose-table:text-foreground prose-th:text-foreground",
            "prose-td:border-border prose-th:border-border",
            isUser 
              ? "bg-gradient-to-br from-chat-bubble-user/50 to-chat-bubble-user/30 border border-border/30 rounded-2xl p-4"
              : ""
          )}>
            {isUser ? (
              <p className="text-sm text-foreground leading-relaxed mb-0">
                {message.content}
              </p>
            ) : (
              <MarkdownRenderer content={message.content} />
            )}
          </div>
        </div>
      </div>

      {/* Subtle border on hover */}
      <div className="absolute inset-0 border border-transparent group-hover:border-border/20 rounded-2xl transition-colors duration-300 pointer-events-none" />
    </div>
  );
};