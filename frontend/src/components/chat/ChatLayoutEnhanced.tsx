import { SidebarProvider, SidebarTrigger, useSidebar } from '@/components/ui/sidebar';
import { ChatSidebarEnhanced } from './ChatSidebarEnhanced';
import { useNavigate } from 'react-router-dom';
import { useIsMobile } from '@/hooks/use-mobile';
import { cn } from '@/lib/utils';
import { Sparkles, Zap } from 'lucide-react';

interface ChatLayoutProps {
  children: React.ReactNode;
}

const ChatLayoutContent = ({ children, onNewChat }: { children: React.ReactNode; onNewChat: () => void }) => {
  const { open } = useSidebar();
  const isMobile = useIsMobile();
  
  return (
    <div className="min-h-screen flex w-full bg-gradient-to-br from-background via-background-subtle to-background-muted">
      <ChatSidebarEnhanced onNewChat={onNewChat} />
      
      <div className={cn(
        "flex-1 flex flex-col transition-all duration-500 ease-[var(--ease-out-expo)]",
        "min-w-0 relative",
        isMobile && "w-full"
      )}>
        {/* Enhanced Header */}
        <header className={cn(
          "h-14 flex items-center justify-between relative z-30",
          "border-b border-border-subtle/50 glass",
          "px-4 sm:px-6 lg:px-8",
          "sticky top-0"
        )}>
          {/* Left Section */}
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <SidebarTrigger className={cn(
              "group relative overflow-hidden",
              "hover:bg-accent-hover rounded-xl p-2.5",
              "transition-all duration-200 ease-[var(--ease-out-cubic)]",
              "hover:scale-110 active:scale-95",
              "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            )}>
              <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-accent/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl" />
            </SidebarTrigger>
            
            <div className="flex items-center gap-3 min-w-0">
              <div className={cn(
                "relative w-8 h-8 rounded-xl flex items-center justify-center",
                "bg-gradient-to-br from-primary/10 to-accent/10",
                "border border-primary/20 shadow-sm",
                "group-hover:shadow-md transition-shadow duration-200"
              )}>
                <Sparkles className="w-4 h-4 text-primary/80" />
                <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </div>
              
              <div className="min-w-0">
                <h1 className={cn(
                  "font-semibold bg-gradient-to-r from-foreground to-foreground-muted bg-clip-text text-transparent",
                  "text-base sm:text-lg tracking-tight truncate"
                )}>
                  Noir Whisper
                </h1>
                <p className="text-xs text-foreground-subtle hidden sm:block">
                  AI Assistant
                </p>
              </div>
            </div>
          </div>
          
          {/* Right Section */}
          <div className="flex items-center gap-3">
            <div className={cn(
              "hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full",
              "bg-gradient-to-r from-accent/20 to-accent/10",
              "border border-accent/30 backdrop-blur-sm"
            )}>
              <div className="relative">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <div className="absolute inset-0 w-2 h-2 bg-green-400/30 rounded-full animate-ping" />
              </div>
              <span className="text-xs text-accent-foreground font-medium">
                Online
              </span>
            </div>
            
            <div className={cn(
              "flex sm:hidden items-center gap-1.5 px-2.5 py-1.5 rounded-full",
              "bg-accent/20 border border-accent/30"
            )}>
              <Zap className="w-3 h-3 text-accent-foreground" />
            </div>
          </div>

          {/* Subtle gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/[0.01] to-transparent pointer-events-none" />
        </header>

        {/* Main Content Area */}
        <main className="flex-1 flex flex-col relative min-h-0 overflow-hidden">
          {/* Ambient Background */}
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-background-subtle/30 to-background-muted/50 pointer-events-none" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/[0.01] via-transparent to-transparent pointer-events-none" />
          
          <div className="flex-1 flex flex-col min-h-0 relative z-10">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export const ChatLayout = ({ children }: ChatLayoutProps) => {
  const navigate = useNavigate();
  const isMobile = useIsMobile();

  const handleNewChat = () => {
    const newChatId = Math.random().toString(36).substr(2, 9);
    navigate(`/chat/${newChatId}`);
  };

  return (
    <SidebarProvider defaultOpen={!isMobile}>
      <ChatLayoutContent onNewChat={handleNewChat}>
        {children}
      </ChatLayoutContent>
    </SidebarProvider>
  );
};