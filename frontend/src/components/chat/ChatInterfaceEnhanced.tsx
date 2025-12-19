import { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { ChatMessageEnhanced } from './ChatMessageEnhanced';
import { ChatInputEnhanced } from './ChatInputEnhanced';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { Sparkles, Code, FileText, Calculator, Paperclip } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  attachments?: File[];
}

// Enhanced mock messages
const mockMessages: Record<string, Message[]> = {
  '1': [
    {
      id: '1',
      role: 'user',
      content: 'Tell me about React hooks and how they work with performance optimization.',
      timestamp: new Date('2024-01-15T10:30:00'),
    },
    {
      id: '2',
      role: 'assistant',
      content: `# React Hooks & Performance Optimization

React Hooks revolutionized how we write components and manage state. Here's a comprehensive guide:

## Core Performance Patterns

### useMemo for Expensive Calculations
\`\`\`javascript
const expensiveValue = useMemo(() => {
  return heavyCalculation(data);
}, [data]);
\`\`\`

### useCallback for Stable References
\`\`\`javascript
const memoizedCallback = useCallback(
  (id) => {
    doSomething(a, b, id);
  },
  [a, b]
);
\`\`\`

## Advanced Optimization Techniques

| Hook | Use Case | Performance Impact |
|------|----------|-------------------|
| useMemo | Heavy computations | High |
| useCallback | Event handlers | Medium |
| React.memo | Component memoization | High |
| useTransition | Non-urgent updates | Medium |

### Mathematical Performance Analysis

The time complexity of re-renders can be expressed as:

$$O(n) = \\sum_{i=1}^{n} C_i \\times R_i$$

Where $C_i$ is the component complexity and $R_i$ is the render frequency.

> **Pro Tip**: Use React DevTools Profiler to identify performance bottlenecks in your component tree.

Would you like me to dive deeper into any specific optimization pattern?`,
      timestamp: new Date('2024-01-15T10:30:05'),
    },
  ],
  '2': [
    {
      id: '3',
      role: 'user',
      content: 'How do I render complex tables and LaTeX equations in markdown?',
      timestamp: new Date('2024-01-14T15:45:00'),
    },
    {
      id: '4',
      role: 'assistant',
      content: `# Advanced Markdown Rendering

Creating rich content with tables, equations, and formatting:

## Complex Tables

\`\`\`markdown
| Feature | Basic | Pro | Enterprise |
|:--------|:-----:|:---:|:----------:|
| Users | 5 | 50 | Unlimited |
| Storage | 1GB | 100GB | 1TB |
| API Calls | 1K/month | 100K/month | Unlimited |
| Support | Email | Priority | Dedicated |
\`\`\`

| Feature | Basic | Pro | Enterprise |
|:--------|:-----:|:---:|:----------:|
| Users | 5 | 50 | Unlimited |
| Storage | 1GB | 100GB | 1TB |
| API Calls | 1K/month | 100K/month | Unlimited |
| Support | Email | Priority | Dedicated |

## Mathematical Expressions

### Inline Math
The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$

### Block Equations
$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$

$$\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}$$

### Matrix Operations
$$\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix} \\begin{pmatrix}
x \\\\
y
\\end{pmatrix} = \\begin{pmatrix}
ax + by \\\\
cx + dy
\\end{pmatrix}$$

These render beautifully with proper LaTeX support!`,
      timestamp: new Date('2024-01-14T15:45:05'),
    },
  ],
};

export const ChatInterface = () => {
  const { chatId } = useParams<{ chatId: string }>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatId && mockMessages[chatId]) {
      setMessages(mockMessages[chatId]);
    } else {
      setMessages([]);
    }
  }, [chatId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (content: string, attachments?: File[]) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
      attachments,
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Simulate AI response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: generateEnhancedResponse(content),
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 1200 + Math.random() * 1800);
  };

  const generateEnhancedResponse = (userInput: string): string => {
    const responses = [
      `# Comprehensive Analysis: "${userInput}"

## Executive Summary

Your question touches on several important concepts that deserve detailed exploration.

### Key Insights

- **Primary Consideration**: Understanding the core principles
- **Secondary Factors**: Implementation strategies and best practices
- **Tertiary Elements**: Edge cases and optimization opportunities

\`\`\`typescript
// Example implementation
interface ResponseStrategy {
  analyze: (input: string) => Analysis;
  synthesize: (data: Analysis[]) => Response;
  optimize: (response: Response) => OptimizedResponse;
}

const strategy: ResponseStrategy = {
  analyze: (input) => ({ /* analysis logic */ }),
  synthesize: (data) => ({ /* synthesis logic */ }),
  optimize: (response) => ({ /* optimization logic */ })
};
\`\`\`

### Mathematical Framework

The relationship can be expressed as:

$$f(x) = \\sum_{i=0}^{n} a_i x^i \\text{ where } a_i \\in \\mathbb{R}$$

### Performance Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Latency | <100ms | 85ms | ✅ |
| Throughput | >1000 RPS | 1250 RPS | ✅ |
| Accuracy | >99% | 99.2% | ✅ |

> **Note**: These metrics are continuously monitored and optimized.

Would you like me to elaborate on any specific aspect?`,

      `# Deep Dive: ${userInput}

## Technical Architecture

Your inquiry requires a multi-layered approach combining theory and practical implementation.

### Core Components

1. **Data Layer**: Foundation for information processing
2. **Logic Layer**: Business rules and validation
3. **Presentation Layer**: User interface and experience

\`\`\`python
# Practical implementation example
class IntelligentProcessor:
    def __init__(self, config):
        self.config = config
        self.optimizer = PerformanceOptimizer()
    
    def process(self, input_data):
        validated = self.validate(input_data)
        processed = self.transform(validated)
        return self.optimize(processed)
    
    def optimize(self, data):
        return self.optimizer.enhance(data)
\`\`\`

### Advanced Concepts

The underlying mathematical model follows:

$$P(success) = \\frac{e^{\\alpha + \\beta x}}{1 + e^{\\alpha + \\beta x}}$$

Where:
- $\\alpha$ represents the baseline probability
- $\\beta$ captures the sensitivity to input $x$

### Comparative Analysis

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| Method A | Fast, Simple | Limited flexibility | Basic scenarios |
| Method B | Highly flexible | Complex setup | Advanced use cases |
| Method C | Balanced | Moderate complexity | General purpose |

This framework provides a solid foundation for your specific needs.`,
    ];

    return responses[Math.floor(Math.random() * responses.length)];
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Messages Area */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="container-fluid max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center min-h-[60vh] sm:min-h-[70vh]">
              <div className="text-center max-w-lg animate-in fade-in-50 slide-in-from-bottom-4 duration-700 space-y-8">
                {/* Hero Icon */}
                <div className="relative mx-auto w-fit">
                  <div className={cn(
                    "w-16 h-16 sm:w-20 sm:h-20 mx-auto rounded-3xl",
                    "bg-gradient-to-br from-primary/20 via-primary/10 to-accent/20",
                    "border border-primary/20 shadow-lg backdrop-blur-sm",
                    "flex items-center justify-center",
                    "animate-in zoom-in-50 duration-500 delay-200"
                  )}>
                    <Sparkles className="w-8 h-8 sm:w-10 sm:h-10 text-primary/80" />
                  </div>
                  
                  {/* Ambient glow */}
                  <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-accent/5 rounded-full blur-2xl scale-150 animate-pulse" />
                </div>

                {/* Welcome Content */}
                <div className="space-y-4">
                  <h2 className={cn(
                    "text-2xl sm:text-3xl font-semibold tracking-tight",
                    "bg-gradient-to-r from-foreground via-foreground-muted to-foreground bg-clip-text text-transparent"
                  )}>
                    Begin Your Conversation
                  </h2>
                  <p className="text-sm sm:text-base text-foreground-subtle leading-relaxed max-w-md mx-auto">
                    Ask me anything about code, math, documentation, or complex problems. 
                    I'll provide detailed, beautifully formatted responses.
                  </p>
                </div>

                {/* Feature Showcase */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
                  {[
                    { icon: Code, label: "Code", color: "from-blue-500/20 to-blue-600/20 border-blue-500/30" },
                    { icon: FileText, label: "Tables", color: "from-green-500/20 to-green-600/20 border-green-500/30" },
                    { icon: Calculator, label: "Math", color: "from-purple-500/20 to-purple-600/20 border-purple-500/30" },
                    { icon: Paperclip, label: "Files", color: "from-orange-500/20 to-orange-600/20 border-orange-500/30" }
                  ].map((feature, index) => (
                    <div 
                      key={feature.label}
                      className={cn(
                        "group p-3 sm:p-4 rounded-2xl border backdrop-blur-sm",
                        "bg-gradient-to-br", feature.color,
                        "hover:scale-105 transition-all duration-300 cursor-default",
                        "animate-in slide-in-from-bottom-2 duration-500"
                      )}
                      style={{ animationDelay: `${400 + index * 100}ms` }}
                    >
                      <feature.icon className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2 text-foreground/70 group-hover:text-foreground/90 transition-colors" />
                      <p className="text-xs sm:text-sm font-medium text-foreground/80 group-hover:text-foreground transition-colors">
                        {feature.label}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="py-4 sm:py-6 space-y-6">
              {messages.map((message) => (
                <ChatMessageEnhanced key={message.id} message={message} />
              ))}
            </div>
          )}
          
          {/* Loading State */}
          {isLoading && (
            <div className={cn(
              "px-4 sm:px-6 py-6 animate-in slide-in-from-bottom-2 duration-300",
              "border-l-2 border-accent/30 bg-gradient-to-r from-accent/5 to-transparent"
            )}>
              <div className="flex gap-4">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent/20 to-accent/10 border border-accent/30 flex items-center justify-center">
                    <div className="w-4 h-4 border-2 border-accent-foreground/30 border-t-accent-foreground rounded-full animate-spin" />
                  </div>
                </div>
                <div className="flex-1 space-y-2">
                  <div className="text-sm font-medium text-foreground">Assistant</div>
                  <div className="flex items-center gap-2">
                    <div className="flex space-x-1">
                      {[0, 1, 2].map((i) => (
                        <div
                          key={i}
                          className="w-1.5 h-1.5 bg-accent-foreground/60 rounded-full animate-bounce"
                          style={{ animationDelay: `${i * 150}ms` }}
                        />
                      ))}
                    </div>
                    <span className="text-xs text-foreground-subtle animate-pulse">
                      Thinking...
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {/* Enhanced Input Area */}
      <ChatInputEnhanced onSendMessage={handleSendMessage} disabled={isLoading} />
    </div>
  );
};