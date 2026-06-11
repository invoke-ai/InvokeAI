import type { FocusEvent, PointerEvent, ReactNode } from 'react';
import { createContext, use, useState } from 'react';

import { useWorkbenchPreferences } from './settings/store';
import type { WidgetRegion } from './types';

interface FocusRegionContextValue {
  focusedRegion: WidgetRegion | null;
  setFocusedRegion: (region: WidgetRegion | null) => void;
}

const FocusRegionContext = createContext<FocusRegionContextValue | null>(null);

const highlightStyles = {
  '&[data-highlighted="true"]::after': {
    borderColor: 'accent.active',
    opacity: 1,
  },
  '&::after': {
    border: '1px solid',
    borderColor: 'transparent',
    borderRadius: 'md',
    content: '""',
    inset: '2px',
    opacity: 0,
    pointerEvents: 'none',
    position: 'absolute',
    transition: 'border-color 0.12s ease, opacity 0.12s ease',
    zIndex: 2,
  },
} as const;

export const FocusRegionProvider = ({ children }: { children: ReactNode }) => {
  const [focusedRegion, setFocusedRegion] = useState<WidgetRegion | null>(null);

  return <FocusRegionContext value={{ focusedRegion, setFocusedRegion }}>{children}</FocusRegionContext>;
};

const useFocusRegionContext = () => {
  const context = use(FocusRegionContext);

  if (!context) {
    throw new Error('useFocusRegionProps must be used within a FocusRegionProvider.');
  }

  return context;
};

export const useFocusRegionProps = (region: WidgetRegion) => {
  const { focusedRegion, setFocusedRegion } = useFocusRegionContext();
  const { showFocusRegionHighlight } = useWorkbenchPreferences();
  const isHighlighted = showFocusRegionHighlight && focusedRegion === region;

  return {
    css: highlightStyles,
    'data-highlighted': isHighlighted,
    onFocusCapture: (_event: FocusEvent<HTMLElement>) => setFocusedRegion(region),
    onPointerDownCapture: (_event: PointerEvent<HTMLElement>) => setFocusedRegion(region),
    position: 'relative' as const,
  };
};
