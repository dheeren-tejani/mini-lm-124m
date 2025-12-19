import { Button } from '../ui/enhanced-button';
import { Textarea } from '@/components/ui/textarea';
import { Send, Paperclip, X, Plus, Smile } from 'lucide-react';
import { useState, useRef, KeyboardEvent } from 'react';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSendMessage: (content: string, attachments?: File[]) => void;
  disabled?: boolean;
}

export const ChatInputEnhanced = ({ onSendMessage, disabled }: ChatInputProps) => {
  const [message, setMessage] = useState('');
  const [attachments, setAttachments] = useState<File[]>([]);
  const [isFocused, setIsFocused] = useState(false);
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

  const canSend = (message.trim() || attachments.length > 0) && !disabled;

  return (
    <div className="glass border-t border-border-subtle/50 p-4 sm:p-6">
      <div className="container-fluid max-w-4xl mx-auto space-y-4">
        {/* Attachment Previews */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 animate-in slide-in-from-bottom-2 duration-300">
            {attachments.map((file, index) => (
              <div
                key={index}
                className={cn(
                  "group flex items-center gap-2 px-3 py-2 rounded-xl text-sm",
                  "bg-gradient-to-r from-accent/20 to-accent/10",
                  "border border-accent/30 text-accent-foreground",
                  "backdrop-blur-sm shadow-sm animate-in scale-in-95 duration-200"
                )}
                style={{ animationDelay: `${index * 50}ms` }}
              >
                <Paperclip className="w-3 h-3 flex-shrink-0" />
                <span className="font-medium truncate max-w-32">
                  {file.name}
                </span>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => removeAttachment(index)}
                  className={cn(
                    "h-5 w-5 ml-1 opacity-60 hover:opacity-100",
                    "hover:bg-accent-foreground/20 hover:scale-110",
                    "transition-all duration-200"
                  )}
                >
                  <X className="w-3 h-3" />
                </Button>
              </div>
            ))}
          </div>
        )}
        
        {/* Main Input Container */}
        <div className={cn(
          "group relative overflow-hidden rounded-3xl transition-all duration-300",
          "bg-gradient-to-br from-card/60 to-card/30 backdrop-blur-xl",
          "border border-border-subtle/50 shadow-lg",
          "hover:shadow-xl hover:border-border/60",
          isFocused && "ring-2 ring-ring/20 ring-offset-2 ring-offset-background border-border/80 shadow-xl"
        )}>
          {/* Ambient Glow Effect */}
          <div className={cn(
            "absolute inset-0 bg-gradient-to-r from-primary/[0.02] via-transparent to-accent/[0.02]",
            "opacity-0 group-hover:opacity-100 transition-opacity duration-500"
          )} />
          
          {/* Input Area */}
          <div className="relative flex items-end gap-3 p-4">
            {/* Action Buttons Left */}
            <div className="flex items-center gap-1 flex-shrink-0">
              {/* File Upload */}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled}
                className={cn(
                  "w-9 h-9 rounded-xl transition-all duration-200",
                  "hover:bg-accent/20 hover:scale-110 active:scale-95",
                  "border border-transparent hover:border-accent/30",
                  "text-foreground-subtle hover:text-foreground"
                )}
              >
                <Paperclip className="w-4 h-4" />
              </Button>

              {/* Add Button */}
              <Button
                variant="ghost"
                size="icon"
                disabled={disabled}
                className={cn(
                  "w-9 h-9 rounded-xl transition-all duration-200",
                  "hover:bg-accent/20 hover:scale-110 active:scale-95",
                  "border border-transparent hover:border-accent/30",
                  "text-foreground-subtle hover:text-foreground"
                )}
              >
                <Plus className="w-4 h-4" />
              </Button>
            </div>
            
            {/* Text Input */}
            <div className="flex-1 relative min-w-0">
              <Textarea
                ref={textareaRef}
                value={message}
                onChange={adjustTextareaHeight}
                onKeyDown={handleKeyDown}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
                placeholder="Message Noir Whisper..."
                disabled={disabled}
                className={cn(
                  "min-h-[2.5rem] max-h-48 resize-none bg-transparent",
                  "border-none outline-none focus:ring-0 focus:ring-offset-0",
                  "text-base placeholder:text-foreground-subtle/60",
                  "leading-relaxed py-2 px-0",
                  "selection:bg-primary/20 selection:text-foreground",
                  "scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border"
                )}
                rows={1}
              />
              
              {/* Character Count */}
              {message.length > 0 && (
                <div className={cn(
                  "absolute bottom-1 right-1 text-2xs text-foreground-subtle/60",
                  "bg-background/80 backdrop-blur-sm rounded px-1.5 py-0.5",
                  "border border-border-subtle/30"
                )}>
                  {message.length}/4000
                </div>
              )}
            </div>
            
            {/* Action Buttons Right */}
            <div className="flex items-center gap-1 flex-shrink-0">
              {/* Emoji Button */}
              {message.length === 0 && (
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={disabled}
                  className={cn(
                    "w-9 h-9 rounded-xl transition-all duration-200",
                    "hover:bg-accent/20 hover:scale-110 active:scale-95",
                    "text-foreground-subtle hover:text-foreground opacity-60 hover:opacity-100"
                  )}
                >
                  <Smile className="w-4 h-4" />
                </Button>
              )}

              {/* Send Button */}
              <Button
                onClick={handleSend}
                disabled={!canSend}
                size="icon"
                className={cn(
                  "w-9 h-9 rounded-xl transition-all duration-200",
                  "shadow-sm hover:shadow-md active:scale-95",
                  canSend
                    ? [
                        "bg-gradient-to-r from-primary to-primary/90",
                        "text-primary-foreground hover:from-primary/90 hover:to-primary/80",
                        "border border-primary/20 hover:border-primary/30",
                        "hover:scale-110 hover:shadow-lg hover:shadow-primary/20"
                      ]
                    : [
                        "bg-muted/50 text-muted-foreground cursor-not-allowed",
                        "border border-muted/30"
                      ]
                )}
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>
          
          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            className="hidden"
            accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.csv,.json"
          />
        </div>
        
        {/* Helper Text */}
        <div className="flex items-center justify-between text-2xs text-foreground-subtle/60">
          <div className="hidden sm:flex items-center gap-4">
            <span>
              Press <kbd className="px-1.5 py-0.5 bg-muted/30 rounded border border-border-subtle/50 font-mono">↵</kbd> to send
            </span>
            <span>
              <kbd className="px-1.5 py-0.5 bg-muted/30 rounded border border-border-subtle/50 font-mono">⇧</kbd> + 
              <kbd className="px-1.5 py-0.5 bg-muted/30 rounded border border-border-subtle/50 font-mono ml-1">↵</kbd> for new line
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span>Powered by AI</span>
            <div className="w-1 h-1 bg-green-400 rounded-full animate-pulse" />
          </div>
        </div>
      </div>
    </div>
  );
};