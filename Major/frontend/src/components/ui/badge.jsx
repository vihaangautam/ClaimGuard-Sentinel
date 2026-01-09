import * as React from "react"
// Wait, I didn't install cva. I'll stick to manual cn for now or install cva.
// User didn't ask for cva. I'll just use cn and simple props.
import { cn } from "@/lib/utils"

const badgeVariants = {
    default: "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
    secondary: "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
    destructive: "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
    outline: "text-foreground",
    success: "border-transparent bg-emerald-500 text-white hover:bg-emerald-600",
    warning: "border-transparent bg-amber-500 text-white hover:bg-amber-600",
}

function Badge({ className, variant = "default", ...props }) {
    const variantClass = badgeVariants[variant] || badgeVariants.default
    return (
        <div className={cn("inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2", variantClass, className)} {...props} />
    )
}

export { Badge, badgeVariants }
