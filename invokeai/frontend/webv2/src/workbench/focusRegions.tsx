import type { WidgetRegion } from '@workbench/layoutContracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { FocusEvent, PointerEvent, ReactNode } from 'react';

import { createContext, use, useState } from 'react';

import { useWorkbenchPreferenceSelector } from './settings/store';

let focusedRegionSnapshot: WidgetRegion | null = null;

export const getFocusedRegionSnapshot = (): WidgetRegion | null => focusedRegionSnapshot;

interface FocusRegionContextValue {
  focusedRegion: WidgetRegion | null;
  setFocusedRegion: (region: WidgetRegion | null) => void;
}

const FocusRegionContext = createContext<FocusRegionContextValue | null>(null);

const highlightStyles = {
  '&[data-highlighted="true"]::after': {
    borderColor: 'accent.solid',
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
    transition: 'border-color var(--wb-motion-duration-fast) ease, opacity var(--wb-motion-duration-fast) ease',
    zIndex: 2,
  },
} as const;

export const FocusRegionProvider = ({ children }: { children: ReactNode }) => {
  const [focusedRegion, setFocusedRegion] = useState<WidgetRegion | null>(null);
  const setFocusedRegionSnapshot = (region: WidgetRegion | null) => {
    focusedRegionSnapshot = region;
    setFocusedRegion(region);
  };

  return (
    <FocusRegionContext value={{ focusedRegion, setFocusedRegion: setFocusedRegionSnapshot }}>
      {children}
    </FocusRegionContext>
  );
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
  const showFocusRegionHighlight = useWorkbenchPreferenceSelector(
    (preferences) => preferences.showFocusRegionHighlight
  );
  const isHighlighted = showFocusRegionHighlight && focusedRegion === region;

  return {
    css: highlightStyles,
    'data-highlighted': isHighlighted,
    onFocusCapture: (_event: FocusEvent<HTMLElement>) => setFocusedRegion(region),
    onPointerDownCapture: (_event: PointerEvent<HTMLElement>) => setFocusedRegion(region),
    position: 'relative' as const,
  };
};
