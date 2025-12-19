import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send, Paperclip, X } from 'lucide-react';
import { useState, useRef, KeyboardEvent } from 'react';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSendMessage: (content: string, attachments?: File[]) => void;
  disabled?: boolean;
}

export const ChatInput = ({ onSendMessage, disabled }: ChatInputProps) => {
  const [message, setMessage] = useState('');
  const [attachments, setAttachments] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleSend = () => {
    if (message.trim() || attachments.length > 0) {
      onSendMessage(message.trim(), attachments);
      setMessage('');
      setAttachments([]);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setAttachments(prev => [...prev, ...files]);
    // trigger border animation
    if (containerRef.current) {
      containerRef.current.classList.remove('animate-border-pulse');
      // Force reflow to restart animation
      // eslint-disable-next-line @typescript-eslint/no-unused-expressions
      containerRef.current.offsetHeight;
      containerRef.current.classList.add('animate-border-pulse');
    }
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="sticky bottom-0 bg-background/40 backdrop-blur-xl p-3 sm:p-6 z-10">
      <div className="max-w-none sm:max-w-4xl mx-auto">
        <div ref={containerRef} className={cn(
          "relative rounded-3xl bg-gradient-to-r from-card/40 to-card/20",
          "backdrop-blur-xl",
          "shadow-lg hover:shadow-xl transition-all duration-300",
          "group border-2 border-border/90",
          attachments.length > 0 && "has-attachments"
        )}>
          {/* Animated top border */}
          <div
            className={cn(
              "pointer-events-none absolute inset-x-0 top-0 h-[2px]",
              "bg-gradient-to-r from-primary/0 via-primary/60 to-accent/60",
              "opacity-0 group-[.has-attachments]:opacity-100 transition-opacity duration-300",
              "rounded-t-3xl"
            )}
          />
          {/* Attachment Previews */}
          {attachments.length > 0 && (
            <div className="p-3 sm:p-4 pb-0">
              <div className="flex flex-wrap gap-2">
                {attachments.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-2 px-2 sm:px-3 py-1.5 sm:py-2 bg-accent/15 rounded-xl shadow-sm animate-scale-in"
                  >
                    <Paperclip className="h-3 w-3 sm:h-4 sm:w-4 text-accent-foreground flex-shrink-0" />
                    <span className="text-xs sm:text-sm text-accent-foreground truncate max-w-20 sm:max-w-32">
                      {file.name}
                    </span>
                    <button
                      onClick={() => removeAttachment(index)}
                      className="ml-1 hover:bg-accent/30 rounded p-0.5 transition-colors flex-shrink-0"
                    >
                      <X className="h-3 w-3 text-accent-foreground/70" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="flex items-center gap-2 sm:gap-3 p-3 sm:p-4">
            {/* File Upload */}
            <div className="flex flex-col gap-2 flex-shrink-0">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                multiple
                className="hidden"
                accept="image/*,text/*,.pdf,.doc,.docx"
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  "h-9 w-10 p-0 rounded-2xl text-xl font-bold",
                  "hover:bg-accent/15 hover:scale-110 transition-all duration-200",
                  "shadow-sm hover:shadow-md",
                  attachments.length > 0 && "bg-accent/20 border border-accent/30"
                )}
              >
                +
              </Button>
            </div>

            {/* Text Input */}
            <div className="flex-1 relative min-w-0">
              <Textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Message Noir Whisper..."
                className={cn(
                  "min-h-[2.25rem] sm:min-h-[2.5rem] max-h-24 sm:max-h-32 resize-none",
                  "bg-transparent border-none focus:ring-0 focus:ring-offset-0",
                  "text-sm sm:text-base placeholder:text-muted-foreground/60",
                  "pr-10 sm:pr-12 py-2"
                )}
                disabled={disabled}
              />

              {/* Send Button */}
              <Button
                onClick={handleSend}
                disabled={disabled || !message.trim()}
                className={cn(
                  "absolute right-1.5 sm:right-2 bottom-1.5 sm:bottom-2 h-7 w-7 sm:h-8 sm:w-8 p-0 rounded-xl",
                  "bg-gradient-to-r from-primary/85 to-accent/90",
                  "hover:from-primary/95 hover:to-accent",
                  "disabled:from-muted disabled:to-muted",
                  "transition-all duration-200 hover:scale-105",
                  "shadow-md hover:shadow-lg hover:shadow-primary/15"
                )}
              >
                <Send className="h-3 w-3 sm:h-4 sm:w-4" />
              </Button>
            </div>
          </div>
        </div>

        {/* Helper Text */}
        <div className="hidden sm:flex items-center justify-between mt-4 text-xs text-muted-foreground/60">
          <span>
            Press <kbd className="px-2 py-1 bg-muted/40 rounded-lg text-xs shadow-sm">Enter</kbd> to send,
            <kbd className="px-2 py-1 bg-muted/40 rounded-lg text-xs ml-1 shadow-sm">Shift+Enter</kbd> for new line
          </span>
          <span>{message.length}/2000</span>
        </div>
      </div>
    </div>
  );
};