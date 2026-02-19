import { PanelLeft, Plus } from 'lucide-react';

interface ChatHeaderProps {
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  onNewChat: () => void;
}

export function ChatHeader({ sidebarOpen, onToggleSidebar, onNewChat }: ChatHeaderProps) {
  return (
    <header className="h-12 flex items-center justify-between px-3 border-b border-border bg-background z-20 shrink-0">
      <div className="flex items-center gap-2">
        <button
          onClick={onToggleSidebar}
          className="w-9 h-9 flex items-center justify-center rounded-md transition-colors duration-200 hover:bg-muted"
          aria-label="Toggle sidebar"
        >
          <PanelLeft className="w-5 h-5 text-muted-foreground" />
        </button>
        <button
          onClick={onNewChat}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-primary text-primary-foreground rounded-full text-sm font-medium transition-colors duration-200 hover:bg-primary/90"
        >
          <Plus className="w-3.5 h-3.5" />
          New chat
        </button>
      </div>

      <div className="flex items-center gap-2">
        <div className="w-7 h-7 rounded-full bg-muted flex items-center justify-center text-xs font-bold text-foreground">
          N
        </div>
        <span className="text-sm font-medium hidden sm:inline">New chat</span>
      </div>

      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-accent" title="Backend connected" />
        <span className="text-xs text-muted-foreground hidden sm:inline">Connected</span>
      </div>
    </header>
  );
}
