import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  [
    "inline-flex items-center justify-center gap-2 whitespace-nowrap",
    "text-sm font-medium transition-all duration-200",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
    "disabled:pointer-events-none disabled:opacity-50",
    "active:scale-[0.98] select-none",
    "[&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0"
  ],
  {
    variants: {
      variant: {
        default: [
          "bg-primary text-primary-foreground shadow-sm",
          "hover:bg-primary-hover hover:shadow-md",
          "active:bg-primary-active"
        ],
        destructive: [
          "bg-destructive text-destructive-foreground shadow-sm",
          "hover:bg-destructive/90 hover:shadow-md"
        ],
        outline: [
          "border border-border bg-transparent shadow-sm",
          "hover:bg-accent hover:text-accent-foreground hover:border-border/80",
          "hover:shadow-md"
        ],
        secondary: [
          "bg-secondary text-secondary-foreground shadow-sm",
          "hover:bg-secondary-hover hover:shadow-md"
        ],
        ghost: [
          "hover:bg-accent hover:text-accent-foreground",
          "hover:shadow-sm"
        ],
        link: [
          "text-primary underline-offset-4",
          "hover:underline hover:text-primary-hover"
        ],
        premium: [
          "bg-gradient-to-r from-primary/10 to-accent/10",
          "border border-primary/20 text-primary-foreground",
          "hover:from-primary/20 hover:to-accent/20 hover:border-primary/30",
          "hover:shadow-lg hover:shadow-primary/10 hover:scale-[1.02]",
          "backdrop-blur-sm"
        ],
        glass: [
          "bg-background/50 backdrop-blur-lg border border-border/50",
          "hover:bg-background/80 hover:border-border/80",
          "hover:shadow-lg"
        ]
      },
      size: {
        default: "h-9 px-4 py-2 rounded-lg",
        sm: "h-8 px-3 text-xs rounded-md",
        lg: "h-10 px-8 rounded-lg",
        xl: "h-12 px-10 text-base rounded-xl",
        icon: "h-9 w-9 rounded-lg",
        "icon-sm": "h-8 w-8 rounded-md",
        "icon-lg": "h-10 w-10 rounded-lg",
        "icon-xl": "h-12 w-12 rounded-xl"
      },
      rounded: {
        none: "rounded-none",
        sm: "rounded-sm",
        md: "rounded-md",
        lg: "rounded-lg",
        xl: "rounded-xl",
        "2xl": "rounded-2xl",
        full: "rounded-full"
      }
    },
    defaultVariants: {
      variant: "default",
      size: "default"
    }
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, rounded, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, rounded, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }