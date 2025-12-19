import { MarkdownRenderer } from './MarkdownRenderer';
import { cn } from '@/lib/utils';
import { User, Bot, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState } from 'react';

interface ChatMessageProps {
  message: {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    attachments?: File[];
  };
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={cn(
      "flex gap-6 p-6 group animate-fade-in-up relative",
      "hover:bg-accent/10 transition-all duration-300 rounded-2xl",
      isUser ? "bg-gradient-to-r from-chat-bubble-user/30 to-transparent" : "bg-gradient-to-r from-chat-bubble-assistant/20 to-transparent"
    )}>
      {/* Avatar */}
      <div className="flex-shrink-0 animate-scale-in">
        <div className={cn(
          "w-10 h-10 rounded-2xl flex items-center justify-center transition-all duration-300",
          "shadow-lg backdrop-blur-sm",
          isUser 
            ? "bg-gradient-to-br from-primary/15 to-primary/8 text-primary" 
            : "bg-gradient-to-br from-secondary/15 to-accent/8 text-secondary-foreground"
        )}>
          {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
        </div>
      </div>
      
      <div className="flex-1 min-w-0 space-y-3">
        {/* Header */}
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-foreground">
            {isUser ? 'You' : 'Assistant'}
          </span>
          <div className="flex items-center gap-2 text-xs text-muted-foreground/60">
            <span className="font-mono">
              {message.timestamp.toLocaleTimeString()}
            </span>
            <div className="w-1 h-1 bg-muted-foreground/40 rounded-full" />
            <span>
              {message.timestamp.toLocaleDateString()}
            </span>
          </div>
        </div>
        
        {/* Attachments */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 animate-fade-in">
            {message.attachments.map((file, index) => (
              <div 
                key={index}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 rounded-full text-sm",
                  "bg-gradient-to-r from-accent/15 to-accent/8",
                  "text-accent-foreground shadow-sm",
                  "hover:from-accent/20 hover:to-accent/12 transition-all duration-200"
                )}
              >
                <span>📎</span>
                <span className="font-medium">{file.name}</span>
              </div>
            ))}
          </div>
        )}
        
        {/* Message Content */}
        <div className="relative">
          <div className={cn(
            "prose prose-invert max-w-none",
            "bg-gradient-to-br from-card/20 to-card/8",
            "rounded-2xl p-4 sm:p-6",
            "backdrop-blur-sm shadow-sm"
          )}>
            <MarkdownRenderer content={message.content} />
          </div>
          
          {/* Copy Button */}
          <Button
            variant="ghost"
            size="sm"
            onClick={copyToClipboard}
            className={cn(
              "absolute -top-2 -right-2 w-8 h-8 p-0 rounded-2xl",
              "bg-background/60 backdrop-blur-sm",
              "opacity-0 group-hover:opacity-100 transition-all duration-300",
              "hover:scale-110 active:scale-95 shadow-md",
              copied && "opacity-100 bg-green-500/10 text-green-400"
            )}
          >
            {copied ? (
              <Check className="w-4 h-4 animate-bounce-in" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};