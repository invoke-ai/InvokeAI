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

const highlightBase = {
  border: '1px solid',
  borderColor: 'transparent',
  borderRadius: 'md',
  content: '""',
  opacity: 0,
  pointerEvents: 'none',
  position: 'absolute',
  transition: 'border-color var(--wb-motion-duration-fast) ease, opacity var(--wb-motion-duration-fast) ease',
  // Above resize handles, so the outline runs through the middle of their grips.
  zIndex: 4,
} as const;

const regionHighlight = (inset: string) => ({
  '&[data-highlighted="true"]::after': { borderColor: 'accent.solid', opacity: 1 },
  '&::after': { ...highlightBase, inset },
});

// Sideways edges sit on the neighbouring divider or rail border rather than beside it. A panel's own divider is
// already its frame's edge; top and bottom stay inside because the workbench row clips vertical overflow.
const HIGHLIGHT_STYLES = {
  bottom: regionHighlight('0 0 -1px 0'),
  center: regionHighlight('0 -1px'),
  left: regionHighlight('0 0 0 -1px'),
  right: regionHighlight('0 -1px 0 0'),
} satisfies Record<WidgetRegion, unknown>;

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

/**
 * The region whose outline is showing, if any. Borders the outline is drawn over hide while it shows, so a shared
 * edge draws one line at any display scale.
 */
export const useHighlightedRegion = (): WidgetRegion | null => {
  const focusedRegion = use(FocusRegionContext)?.focusedRegion ?? null;
  const showFocusRegionHighlight = useWorkbenchPreferenceSelector(
    (preferences) => preferences.showFocusRegionHighlight
  );
  return showFocusRegionHighlight ? focusedRegion : null;
};

export const useFocusRegionProps = (region: WidgetRegion) => {
  const { setFocusedRegion } = useFocusRegionContext();
  const isHighlighted = useHighlightedRegion() === region;

  return {
    css: HIGHLIGHT_STYLES[region],
    'data-highlighted': isHighlighted,
    onFocusCapture: (_event: FocusEvent<HTMLElement>) => setFocusedRegion(region),
    onPointerDownCapture: (_event: PointerEvent<HTMLElement>) => setFocusedRegion(region),
    position: 'relative' as const,
  };
};
