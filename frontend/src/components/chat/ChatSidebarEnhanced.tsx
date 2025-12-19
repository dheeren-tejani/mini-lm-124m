import { useState } from 'react';
import { MessageSquare, Plus, History, Settings, MoreHorizontal, Edit3, Trash2, Clock } from 'lucide-react';
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
import { Button } from '../ui/enhanced-button';
import { cn } from '@/lib/utils';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface ChatSidebarProps {
  onNewChat: () => void;
}

export const ChatSidebarEnhanced = ({ onNewChat }: ChatSidebarProps) => {
  const navigate = useNavigate();
  const { chatId } = useParams<{ chatId: string }>(); 
  const { state, setOpen } = useSidebar();
  const isMobile = useIsMobile();
  const [chats] = useState([
    { 
      id: '1', 
      title: 'React Hooks & Performance', 
      timestamp: '2h ago', 
      preview: 'Tell me about React hooks and optimization...',
      unread: false,
      category: 'Development'
    },
    { 
      id: '2', 
      title: 'Markdown & LaTeX Rendering', 
      timestamp: '1d ago', 
      preview: 'How do I render complex tables and equations...',
      unread: true,
      category: 'Documentation'
    },
    { 
      id: '3', 
      title: 'TypeScript Advanced Patterns', 
      timestamp: '3d ago', 
      preview: 'What are some advanced TypeScript patterns...',
      unread: false,
      category: 'Development'
    },
    { 
      id: '4', 
      title: 'State Management Architecture', 
      timestamp: '1w ago', 
      preview: 'Comparing Redux, Zustand, and Context API...',
      unread: false,
      category: 'Architecture'
    },
  ]);

  const handleChatNavigation = (chatId: string) => {
    navigate(`/chat/${chatId}`);
    if (isMobile) {
      setOpen(false);
    }
  };

  const isCollapsed = state === "collapsed";

  return (
    <>
      {/* Mobile Backdrop */}
      {isMobile && !isCollapsed && (
        <div 
          className="fixed inset-0 bg-background/60 backdrop-blur-sm z-40 lg:hidden animate-in fade-in-0 duration-300"
          onClick={() => setOpen(false)}
        />
      )}

      <Sidebar className={cn(
        "border-r border-sidebar-border glass transition-all duration-500 ease-[var(--ease-out-expo)]",
        "flex-shrink-0 relative z-50",
        isCollapsed ? "w-14" : "w-72",
        isMobile && isCollapsed && "w-0 border-r-0 overflow-hidden",
        isMobile && !isCollapsed && "fixed inset-y-0 left-0 w-72 shadow-2xl"
      )}>
        <SidebarContent className={cn(
          "flex flex-col h-full overflow-hidden",
          "p-3 space-y-4"
        )}>
          {/* Header Section */}
          <div className="flex-shrink-0 space-y-3">
            <Button
              onClick={onNewChat}
              variant="premium"
              size={isCollapsed ? "icon" : "default"}
              className={cn(
                "w-full group relative overflow-hidden",
                "transition-all duration-300 ease-[var(--ease-out-expo)]",
                "hover:scale-[1.02] active:scale-[0.98]",
                "shadow-lg hover:shadow-xl",
                isCollapsed && "aspect-square"
              )}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-transparent to-accent/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <Plus className={cn(
                "transition-transform duration-200 group-hover:rotate-90",
                "w-4 h-4", 
                !isCollapsed && "mr-2"
              )} />
              {!isCollapsed && (
                <span className="font-medium tracking-wide">New Chat</span>
              )}
            </Button>

            {!isCollapsed && (
              <div className="px-1">
                <div className="flex items-center gap-2 text-xs text-foreground-subtle">
                  <History className="w-3 h-3" />
                  <span className="font-medium uppercase tracking-wider">Recent</span>
                </div>
              </div>
            )}
          </div>

          {/* Chat History */}
          <SidebarGroup className="flex-1 min-h-0 overflow-hidden">
            <SidebarGroupContent className="h-full">
              <SidebarMenu className="space-y-1 overflow-y-auto scrollbar-thin">
                {chats.map((chat, index) => (
                  <SidebarMenuItem key={chat.id}>
                    <div 
                      className={cn(
                        "group relative animate-in slide-in-from-left-1 duration-300",
                        "hover:bg-sidebar-hover rounded-xl transition-all duration-200",
                        chatId === chat.id && "bg-gradient-to-r from-sidebar-accent/20 to-sidebar-accent/10 border border-sidebar-accent/30"
                      )}
                      style={{ animationDelay: `${index * 50}ms` }}
                    >
                      <SidebarMenuButton 
                        asChild
                        className={cn(
                          "w-full p-0 h-auto hover:bg-transparent",
                          "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1",
                          "rounded-xl"
                        )}
                      >
                        <button 
                          onClick={() => handleChatNavigation(chat.id)}
                          className={cn(
                            "w-full text-left p-3 flex items-start gap-3",
                            "transition-all duration-200 ease-[var(--ease-out-cubic)]",
                            "hover:scale-[1.01] active:scale-[0.99]",
                            "rounded-xl relative overflow-hidden",
                            isCollapsed && "justify-center p-2.5"
                          )}
                        >
                          {/* Icon */}
                          <div className={cn(
                            "flex-shrink-0 relative",
                            "w-8 h-8 rounded-lg flex items-center justify-center",
                            "bg-gradient-to-br from-sidebar-accent/10 to-sidebar-accent/5",
                            "border border-sidebar-accent/20",
                            "transition-all duration-200 group-hover:border-sidebar-accent/40",
                            isCollapsed && "w-6 h-6"
                          )}>
                            <MessageSquare className={cn(
                              "text-sidebar-accent-foreground/70",
                              isCollapsed ? "w-3 h-3" : "w-4 h-4"
                            )} />
                            {chat.unread && (
                              <div className="absolute -top-1 -right-1 w-2 h-2 bg-blue-500 rounded-full border border-sidebar-background" />
                            )}
                          </div>
                          
                          {/* Content */}
                          {!isCollapsed && (
                            <div className="flex-1 min-w-0 space-y-1">
                              <div className="flex items-start justify-between gap-2">
                                <h4 className={cn(
                                  "font-medium text-sm text-sidebar-foreground truncate-2",
                                  "leading-tight"
                                )}>
                                  {chat.title}
                                </h4>
                                <div className="flex items-center gap-1 flex-shrink-0">
                                  <Clock className="w-3 h-3 text-sidebar-foreground/40" />
                                  <span className="text-2xs text-sidebar-foreground/40 font-mono">
                                    {chat.timestamp}
                                  </span>
                                </div>
                              </div>
                              
                              <p className="text-xs text-sidebar-foreground/60 truncate-1 leading-relaxed">
                                {chat.preview}
                              </p>
                              
                              <div className="flex items-center justify-between">
                                <span className={cn(
                                  "text-2xs px-1.5 py-0.5 rounded-md font-medium",
                                  "bg-sidebar-accent/10 text-sidebar-accent-foreground/70",
                                  "border border-sidebar-accent/20"
                                )}>
                                  {chat.category}
                                </span>
                                {chat.unread && (
                                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full" />
                                )}
                              </div>
                            </div>
                          )}
                          
                          {/* Hover Effect */}
                          <div className="absolute inset-0 bg-gradient-to-r from-sidebar-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none rounded-xl" />
                        </button>
                      </SidebarMenuButton>

                      {/* Actions Menu */}
                      {!isCollapsed && (
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon-sm"
                              className={cn(
                                "absolute top-2 right-2 h-6 w-6",
                                "opacity-0 group-hover:opacity-100 transition-all duration-200",
                                "hover:bg-sidebar-accent/20 hover:scale-110",
                                "focus-visible:opacity-100"
                              )}
                            >
                              <MoreHorizontal className="w-3 h-3" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent 
                            align="end" 
                            className="glass border-border/50 animate-in slide-in-from-top-1 duration-200"
                          >
                            <DropdownMenuItem className="hover:bg-accent/50 focus:bg-accent/50">
                              <Edit3 className="w-4 h-4 mr-2" />
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem className="text-destructive hover:bg-destructive/10 focus:bg-destructive/10">
                              <Trash2 className="w-4 h-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      )}
                    </div>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>

          {/* Footer */}
          <div className="flex-shrink-0 pt-3 border-t border-sidebar-border/50">
            <Button
              variant="ghost"
              size={isCollapsed ? "icon" : "default"}
              className={cn(
                "w-full text-sidebar-foreground/70 hover:text-sidebar-foreground",
                "hover:bg-sidebar-hover transition-all duration-200",
                "group relative overflow-hidden"
              )}
            >
              <Settings className={cn(
                "transition-transform duration-200 group-hover:rotate-90",
                "w-4 h-4",
                !isCollapsed && "mr-2"
              )} />
              {!isCollapsed && (
                <span className="font-medium">Settings</span>
              )}
              <div className="absolute inset-0 bg-gradient-to-r from-sidebar-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none rounded-lg" />
            </Button>
          </div>
        </SidebarContent>
      </Sidebar>
    </>
  );
};