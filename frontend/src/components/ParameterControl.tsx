import { useState, useRef } from 'react';
import { Info } from 'lucide-react';

interface ParameterControlProps {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  type?: 'slider' | 'number';
}

export function ParameterControl({
  label,
  description,
  value,
  min,
  max,
  step,
  onChange,
  type = 'number',
}: ParameterControlProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const handleChange = (val: string) => {
    const num = parseFloat(val);
    if (!isNaN(num)) {
      onChange(Math.min(max, Math.max(min, num)));
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 relative">
          <span className="text-xs font-medium text-foreground">{label}</span>
          <button
            className="text-muted-foreground hover:text-foreground transition-colors duration-200"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            onFocus={() => setShowTooltip(true)}
            onBlur={() => setShowTooltip(false)}
            aria-label={`Info about ${label}`}
          >
            <Info className="w-3 h-3" />
          </button>
          {showTooltip && (
            <div
              ref={tooltipRef}
              className="absolute left-0 top-full mt-1 z-50 px-2.5 py-1.5 bg-popover border border-border rounded-md text-xs text-popover-foreground max-w-[220px] animate-message-in"
            >
              {description}
            </div>
          )}
        </div>
        <span className="text-xs text-muted-foreground tabular-nums">{step < 1 ? value.toFixed(2) : value}</span>
      </div>

      {type === 'slider' ? (
        <div className="flex items-center gap-2">
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={e => handleChange(e.target.value)}
            className="flex-1 h-1 accent-accent bg-muted rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-foreground [&::-webkit-slider-thumb]:cursor-pointer"
          />
        </div>
      ) : (
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => handleChange(e.target.value)}
          className="w-full h-8 px-2.5 text-xs bg-card border border-border rounded-md text-foreground outline-none transition-shadow duration-200 focus:shadow-[0_0_0_1px_hsl(0_0%_20%)] tabular-nums"
        />
      )}
    </div>
  );
}
