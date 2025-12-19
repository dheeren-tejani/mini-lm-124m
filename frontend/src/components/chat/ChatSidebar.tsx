import { useEffect, useRef, useState } from 'react';
import { MessageSquare, Plus, History, Settings, MoreVertical, Edit3, Trash2, Download } from 'lucide-react';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { useNavigate, useParams } from 'react-router-dom';
import { useIsMobile } from '@/hooks/use-mobile';
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar
} from '@/components/ui/sidebar';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { StoredChatMeta, getChats, subscribe, setChats } from '@/lib/chatStore';

interface ChatSidebarProps {
  onNewChat: () => void;
}

export const ChatSidebar = ({ onNewChat }: ChatSidebarProps) => {
  const navigate = useNavigate();
  const { chatId } = useParams<{ chatId: string }>();
  const { state, setOpen } = useSidebar();
  const isMobile = useIsMobile();
  const [chats, setChatsState] = useState<StoredChatMeta[]>([]);
  const dragIndexRef = useRef<number | null>(null);

  useEffect(() => {
    setChatsState(getChats());
    const unsub = subscribe(() => setChatsState(getChats()));
    return unsub;
  }, []);

  // Auto-collapse sidebar on mobile when navigating to a chat
  const handleChatNavigation = (chatId: string) => {
    navigate(`/chat/${chatId}`);
    if (isMobile) {
      setOpen(false);
    }
  };

  const isCollapsed = state === "collapsed";

  return (
    <Sidebar className={cn(
      "bg-sidebar-background/60 backdrop-blur-xl",
      "transition-all duration-500 ease-in-out",
      "flex-shrink-0",
      isCollapsed ? "w-16" : "w-80",
      isMobile && isCollapsed && "w-0 overflow-hidden",
      isMobile && !isCollapsed && "fixed inset-y-0 left-0 z-50 w-80 shadow-2xl"
    )}>
      <SidebarContent className={cn(
        "p-3 sm:p-4 space-y-4 sm:space-y-6",
        "flex flex-col h-full"
      )}>
        {/* Mobile overlay */}
        {isMobile && !isCollapsed && (
          <div
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 lg:hidden"
            onClick={() => setOpen(false)}
          />
        )}

        {/* New Chat Button */}
        <div className="space-y-3 flex-shrink-0 mt-16">
          <Button
            onClick={onNewChat}
            className={cn(
              "w-full bg-white from-primary/8 to-accent/8",
              "text-primary-foreground backdrop-blur-sm",
              "hover:from-primary/15 hover:to-accent/15",
              "transition-all duration-300 hover:scale-[1.02] hover:shadow-lg hover:shadow-primary/5",
              "rounded-2xl font-medium text-sm sm:text-base shadow-sm",
              isCollapsed && "w-12 h-12 p-0"
            )}
          >
            <Plus className={cn("h-4 w-4 sm:h-5 sm:w-5", !isCollapsed && "mr-2")} />
            {!isCollapsed && "New Chat"}
          </Button>
        </div>

        {/* Chat History */}
        <SidebarGroup className="flex-1 min-h-0">
          <div className={cn(
            "flex items-center gap-2 mb-3 sm:mb-4",
            isCollapsed && "justify-center"
          )}>
            <History className="h-4 w-4 text-muted-foreground flex-shrink-0" />
            {!isCollapsed && (
              <span className="text-sm font-medium text-muted-foreground">Recent Chats</span>
            )}
          </div>

          <SidebarGroupContent className="flex-1 min-h-0">
            <SidebarMenu className="space-y-1 sm:space-y-2 overflow-y-auto">
              {chats.map((chat, index) => (
                <SidebarMenuItem
                  key={chat.id}
                  draggable
                  onDragStart={(e) => {
                    dragIndexRef.current = index;
                    e.dataTransfer.effectAllowed = 'move';
                  }}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const from = dragIndexRef.current;
                    if (from === null) return;
                    const to = index;
                    if (from === to) return;
                    const next = [...chats];
                    const [moved] = next.splice(from, 1);
                    next.splice(to, 0, moved);
                    setChats(next); // persist to localStorage
                    setChatsState(next); // update local UI
                    dragIndexRef.current = null;
                  }}
                >
                  <div className={cn(
                    "group relative overflow-hidden rounded-2xl",
                    "hover:bg-gradient-to-r hover:from-accent/8 hover:to-accent/4",
                    "transition-all duration-300",
                    chatId === chat.id && "bg-gradient-to-r from-accent/15 to-accent/8 shadow-sm",
                    isCollapsed && "w-12 h-12 p-0 justify-center"
                  )}>
                    <div className="flex items-center">
                      <button
                        onClick={() => handleChatNavigation(chat.id)}
                        className={cn(
                          "flex-1 text-left p-2 sm:p-3 flex items-start gap-2 sm:gap-3",
                          isCollapsed && "justify-center p-0"
                        )}
                      >
                        {/* Numbered badge */}
                        <div className="flex-shrink-0 w-6 h-6 rounded-md bg-accent/15 border border-accent/30 flex items-center justify-center">
                          <span className="text-xs font-extrabold text-sidebar-foreground">{index + 1}.</span>
                        </div>

                        {!isCollapsed && (
                          <div className="flex-1 min-w-0">
                            <div className="font-semibold text-sm text-sidebar-foreground/90 truncate-2">
                              {chat.title}
                            </div>
                          </div>
                        )}
                      </button>

                      {/* Three dots menu */}
                      {!isCollapsed && (
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className={cn(
                                "h-8 w-8 p-0 rounded-lg opacity-0 group-hover:opacity-100",
                                "transition-all duration-200 hover:bg-accent/20 mr-2"
                              )}
                              onClick={(e) => e.stopPropagation()}
                            >
                              <MoreVertical className="h-4 w-4 text-sidebar-foreground/70" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="w-48">
                            <DropdownMenuItem className="flex items-center gap-2">
                              <Edit3 className="h-4 w-4" />
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem className="flex items-center gap-2">
                              <Download className="h-4 w-4" />
                              Export
                            </DropdownMenuItem>
                            <DropdownMenuItem className="flex items-center gap-2 text-destructive">
                              <Trash2 className="h-4 w-4" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      )}
                    </div>
                  </div>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Settings */}
        <div className="mt-auto pt-3 sm:pt-4 flex-shrink-0">
          <Button
            variant="ghost"
            className={cn(
              "w-full justify-start text-sidebar-foreground/70 hover:text-sidebar-foreground",
              "hover:bg-accent/8 rounded-2xl transition-all duration-200 text-sm",
              isCollapsed && "w-12 h-12 p-0 justify-center"
            )}
          >
            <Settings className={cn("h-4 w-4", !isCollapsed && "mr-2")} />
            {!isCollapsed && "Settings"}
          </Button>
        </div>
      </SidebarContent>
    </Sidebar>
  );
};