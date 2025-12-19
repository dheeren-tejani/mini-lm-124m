import { SidebarProvider, SidebarTrigger, useSidebar } from '@/components/ui/sidebar';
import { ChatSidebar } from './ChatSidebar';
import { useNavigate, useParams } from 'react-router-dom';
import { useIsMobile } from '@/hooks/use-mobile';
import { cn } from '@/lib/utils';
import { createChat, getChats } from '@/lib/chatStore';

interface ChatLayoutProps {
  children: React.ReactNode;
}

const ChatLayoutContent = ({ children, onNewChat }: { children: React.ReactNode; onNewChat: () => void }) => {
  const { open } = useSidebar();
  const isMobile = useIsMobile();
  const { chatId } = useParams<{ chatId: string }>();

  // Get current chat title
  const currentChat = chatId ? getChats().find(chat => chat.id === chatId) : null;
  const chatTitle = currentChat?.title || 'Noir Whisper';

  return (
    <div className="min-h-screen flex w-full bg-gradient-to-br from-background via-background/95 to-background">
      <ChatSidebar onNewChat={onNewChat} />

      <div className={cn(
        "flex-1 flex flex-col transition-all duration-500 ease-in-out",
        "min-w-0", // Prevent flex item from overflowing
        isMobile && "w-full"
      )}>
        {/* Header */}
        <header className={cn(
          "h-16 flex items-center justify-between",
          "bg-background/60 backdrop-blur-xl",
          "px-3 sm:px-6",
          "sticky top-0 z-40"
        )}>
          <div className="flex items-center gap-2 sm:gap-4 min-w-0">
            <SidebarTrigger className={cn(
              "hover:bg-accent/50 rounded-xl p-2.5 transition-all duration-200",
              "hover:scale-110 active:scale-95 flex-shrink-0"
            )} />
            <div className="flex items-center gap-2 sm:gap-3 min-w-0">
              <div className="w-8 h-8 bg-gradient-to-br from-primary/15 to-accent/15 rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg">
                <span className="text-sm font-bold">N</span>
              </div>
              <h1 className={cn(
                "font-bold bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent",
                "text-lg sm:text-xl truncate"
              )}>
                {chatTitle}
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-4">
            <div className={cn(
              "flex items-center gap-2 px-3 sm:px-4 py-2 rounded-full",
              "bg-accent/15 backdrop-blur-sm shadow-sm"
            )}>
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse flex-shrink-0"></div>
              <span className="text-xs sm:text-sm text-muted-foreground font-medium hidden xs:inline">
                AI Assistant
              </span>
            </div>
          </div>
        </header>

        {/* Main Chat Area */}
        <main className="flex-1 flex flex-col relative min-h-0">
          <div className="flex-1 flex flex-col min-h-0">
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
    // Generate a new chat ID and navigate to it
    const newChatId = Math.random().toString(36).substr(2, 9);
    createChat(newChatId);
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