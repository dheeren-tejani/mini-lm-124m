import { Button } from '@/components/ui/button';
import { Plus, ArrowUp, Paperclip, X } from 'lucide-react';
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
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (message.trim() || attachments.length > 0) {
      onSendMessage(message.trim(), attachments);
      setMessage('');
      setAttachments([]);
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
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
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const adjustTextareaHeight = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = e.target;
    setMessage(textarea.value);
    
    textarea.style.height = 'auto';
    const maxHeight = 200;
    const newHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = `${newHeight}px`;
  };

  return (
    <div className="bg-gradient-to-t from-background/95 to-background/80 backdrop-blur-xl p-6 pb-8 border-t border-border/30">
      <div className="max-w-4xl mx-auto space-y-4">
        {/* Attachments Preview */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 animate-fade-in-up">
            {attachments.map((file, index) => (
              <div 
                key={index}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-xl text-sm",
                  "bg-gradient-to-r from-accent/20 to-accent/10",
                  "border border-accent/30 text-accent-foreground",
                  "backdrop-blur-sm shadow-sm animate-scale-in"
                )}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <span>📎</span>
                <span className="font-medium">{file.name}</span>
                <button
                  onClick={() => removeAttachment(index)}
                  className={cn(
                    "hover:bg-accent-foreground/20 rounded-full p-1",
                    "transition-all duration-200 hover:scale-110 active:scale-95"
                  )}
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}
        
        {/* Input Container */}
        <div className={cn(
          "relative group animate-fade-in-up",
          "bg-gradient-to-br from-card/80 to-card/40",
          "border border-border/50 rounded-3xl shadow-xl",
          "backdrop-blur-xl overflow-hidden",
          "hover:shadow-2xl hover:border-border/80 transition-all duration-300"
        )}>
          {/* Glow Effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-accent/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          
          <div className="relative flex items-end gap-3 p-4">
            {/* Attach Button */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                "flex-shrink-0 w-11 h-11 rounded-2xl",
                "hover:bg-accent/50 transition-all duration-200",
                "hover:scale-110 active:scale-95",
                "bg-gradient-to-br from-accent/10 to-accent/5",
                "border border-accent/20"
              )}
              disabled={disabled}
            >
              <Plus className="w-5 h-5" />
            </Button>
            
            {/* Text Input */}
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={message}
                onChange={adjustTextareaHeight}
                onKeyDown={handleKeyDown}
                placeholder="How can I help you today?"
                disabled={disabled}
                className={cn(
                  "w-full min-h-[28px] max-h-[200px] resize-none bg-transparent",
                  "border-none outline-none text-foreground placeholder:text-muted-foreground/60",
                  "text-base leading-7 py-2 px-1",
                  "selection:bg-primary/20"
                )}
                rows={1}
              />
            </div>
            
            {/* Send Button */}
            <Button
              onClick={handleSend}
              disabled={disabled || (!message.trim() && attachments.length === 0)}
              size="sm"
              className={cn(
                "flex-shrink-0 w-11 h-11 rounded-2xl transition-all duration-300",
                "hover:scale-110 active:scale-95",
                (!message.trim() && attachments.length === 0) 
                  ? "bg-muted/50 text-muted-foreground cursor-not-allowed border border-muted/30" 
                  : [
                      "bg-gradient-to-br from-foreground to-foreground/80",
                      "text-background hover:from-foreground/90 hover:to-foreground/70",
                      "shadow-lg hover:shadow-xl border border-foreground/20",
                      "hover:animate-pulse-glow"
                    ]
              )}
            >
              <ArrowUp className="w-5 h-5" />
            </Button>
          </div>
          
          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            className="hidden"
            accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt"
          />
        </div>
        
        {/* Helper Text */}
        <div className="text-center">
          <p className="text-xs text-muted-foreground/60 font-mono">
            Press <kbd className="px-1.5 py-0.5 bg-muted/30 rounded border border-border/50">Enter</kbd> to send, 
            <kbd className="px-1.5 py-0.5 bg-muted/30 rounded border border-border/50 ml-1">Shift + Enter</kbd> for new line
          </p>
        </div>
      </div>
    </div>
  );
};