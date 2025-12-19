import { Button } from '@/components/ui/button';
import { MessageSquare, Sparkles, Code, Table } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Index = () => {
  const navigate = useNavigate();

  const handleStartChat = () => {
    navigate('/chat');
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <div className="flex flex-col items-center justify-center min-h-screen px-4">
        <div className="text-center max-w-3xl mx-auto">
          <div className="mb-8">
            <h1 className="text-5xl font-bold text-foreground mb-4">
              Noir Whisper
            </h1>
            <p className="text-xl text-muted-foreground mb-8">
              An elegant AI chatbot with advanced markdown rendering, multimodal input, and persistent chat history
            </p>
          </div>

          <Button 
            onClick={handleStartChat}
            size="lg"
            className="mb-16 px-8 py-6 text-lg"
          >
            <MessageSquare className="w-5 h-5 mr-2" />
            Start Chatting
          </Button>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="bg-card text-card-foreground p-6 rounded-lg border border-border">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <Code className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Code Highlighting</h3>
              <p className="text-muted-foreground text-sm">
                Beautiful syntax highlighting for multiple programming languages with copy functionality
              </p>
            </div>

            <div className="bg-card text-card-foreground p-6 rounded-lg border border-border">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <Table className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Rich Markdown</h3>
              <p className="text-muted-foreground text-sm">
                Full markdown support including tables, lists, quotes, and formatted text rendering
              </p>
            </div>

            <div className="bg-card text-card-foreground p-6 rounded-lg border border-border">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <Sparkles className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Multimodal Input</h3>
              <p className="text-muted-foreground text-sm">
                Upload images, documents, and files alongside your text messages for richer conversations
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
