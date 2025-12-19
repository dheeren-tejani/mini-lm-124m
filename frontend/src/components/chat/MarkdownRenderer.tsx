import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import 'katex/dist/katex.min.css';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Copy, Check } from 'lucide-react';
import { useState } from 'react';

// Color swatch component
const ColorSwatch = ({ hex }: { hex: string }) => {
  return (
    <span className="inline-flex items-center gap-2">
      <div
        className="w-4 h-4 rounded border border-border/30 shadow-sm flex-shrink-0"
        style={{ backgroundColor: hex }}
      />
      <code className="bg-gradient-to-r from-accent/20 to-accent/10 border border-accent/30 px-1.5 py-0.5 rounded text-sm font-mono text-accent-foreground">
        {hex}
      </code>
    </span>
  );
};

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export const MarkdownRenderer = ({ content, className }: MarkdownRendererProps) => {
  const [copiedStates, setCopiedStates] = useState<{ [key: string]: boolean }>({});

  const handleCopyCode = async (code: string, blockId: string) => {
    await navigator.clipboard.writeText(code);
    setCopiedStates(prev => ({ ...prev, [blockId]: true }));
    setTimeout(() => {
      setCopiedStates(prev => ({ ...prev, [blockId]: false }));
    }, 2000);
  };

  // Function to process text and replace hex codes with color swatches
  const processTextWithColorSwatches = (text: string) => {
    // Regex to match hex codes (6 or 3 digit hex codes)
    const hexRegex = /#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})\b/g;

    const parts = text.split(hexRegex);
    const matches = text.match(hexRegex) || [];

    if (matches.length === 0) return text;

    const result = [];
    for (let i = 0; i < parts.length; i++) {
      result.push(parts[i]);
      if (i < matches.length) {
        result.push(<ColorSwatch key={`color-${i}`} hex={matches[i]} />);
      }
    }

    return result;
  };

  return (
    <div className={cn("prose prose-invert max-w-none", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code({ className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';
            const isCodeBlock = match && className?.includes('language-');
            const codeContent = String(children).replace(/\n$/, '');
            const blockId = `code-${Math.random().toString(36).substr(2, 9)}`;
            const isCopied = copiedStates[blockId] || false;

            return isCodeBlock ? (
              <div className="relative group">
                <div className="flex items-center justify-between bg-background/50 border border-border/30 rounded-t-lg px-3 py-2">
                  <span className="text-xs text-muted-foreground font-medium uppercase">
                    {language}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleCopyCode(codeContent, blockId)}
                    className={cn(
                      "h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity",
                      isCopied && "opacity-100"
                    )}
                  >
                    {isCopied ? (
                      <Check className="h-3 w-3 text-green-400" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </Button>
                </div>
                <SyntaxHighlighter
                  style={vscDarkPlus}
                  language={language}
                  PreTag="div"
                  className="!mt-0 !rounded-t-none"
                  customStyle={{
                    margin: '0',
                    backgroundColor: 'hsl(var(--background))',
                    border: '1px solid hsl(var(--border) / 0.3)',
                    borderTop: 'none',
                    fontSize: '0.875rem',
                    lineHeight: '1.5',
                  } as any}
                  showLineNumbers={language && ['javascript', 'typescript', 'python', 'java', 'cpp', 'c', 'go', 'rust'].includes(language)}
                  wrapLines={true}
                >
                  {codeContent}
                </SyntaxHighlighter>
              </div>
            ) : (
              <code
                className="bg-gradient-to-r from-accent/20 to-accent/10 border border-accent/30 px-1.5 py-0.5 rounded text-sm font-mono text-accent-foreground"
                {...props}
              >
                {children}
              </code>
            );
          },
          table({ children }) {
            return (
              <div className="overflow-x-auto my-4">
                <table className="min-w-full border border-border/50 rounded-lg shadow-sm">
                  {children}
                </table>
              </div>
            );
          },
          thead({ children }) {
            return (
              <thead className="bg-gradient-to-r from-accent/10 to-accent/5">
                {children}
              </thead>
            );
          },
          th({ children }) {
            return (
              <th className="border border-border/50 px-4 py-3 text-left font-semibold text-accent-foreground">
                {children}
              </th>
            );
          },
          td({ children }) {
            return (
              <td className="border border-border/50 px-4 py-3 text-foreground">
                {children}
              </td>
            );
          },
          h1({ children }) {
            return (
              <h1 className="text-2xl font-bold mb-4 bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent">
                {children}
              </h1>
            );
          },
          h2({ children }) {
            return (
              <h2 className="text-xl font-semibold mb-3 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                {children}
              </h2>
            );
          },
          h3({ children }) {
            return (
              <h3 className="text-lg font-medium mb-2 text-accent-foreground">
                {children}
              </h3>
            );
          },
          p({ children }) {
            return (
              <p className="mb-3 text-foreground leading-relaxed">
                {typeof children === 'string' ? processTextWithColorSwatches(children) : children}
              </p>
            );
          },
          ul({ children }) {
            return (
              <ul className="list-disc pl-6 mb-3 text-foreground">
                {children}
              </ul>
            );
          },
          ol({ children }) {
            return (
              <ol className="list-decimal pl-6 mb-3 text-foreground">
                {children}
              </ol>
            );
          },
          li({ children }) {
            return (
              <li className="mb-1">
                {typeof children === 'string' ? processTextWithColorSwatches(children) : children}
              </li>
            );
          },
          blockquote({ children }) {
            return (
              <blockquote className="border-l-4 border-gradient-to-b from-primary to-accent bg-gradient-to-r from-accent/5 to-transparent pl-4 my-4 italic text-muted-foreground rounded-r-lg py-2">
                {typeof children === 'string' ? processTextWithColorSwatches(children) : children}
              </blockquote>
            );
          },
          strong({ children }) {
            return (
              <strong className="font-bold text-accent-foreground">
                {typeof children === 'string' ? processTextWithColorSwatches(children) : children}
              </strong>
            );
          },
          em({ children }) {
            return (
              <em className="italic text-primary">
                {typeof children === 'string' ? processTextWithColorSwatches(children) : children}
              </em>
            );
          },
          a({ href, children }) {
            return (
              <a
                href={href}
                className="text-primary hover:text-accent underline decoration-primary/50 hover:decoration-accent transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};