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
    'data-focus-region': region,
    'data-highlighted': isHighlighted,
    onFocusCapture: (_event: FocusEvent<HTMLElement>) => setFocusedRegion(region),
    onPointerDownCapture: (_event: PointerEvent<HTMLElement>) => setFocusedRegion(region),
    position: 'relative' as const,
  };
};

/** A lazy widget mounts within a few frames of opening; stop looking after about half a second. */
const OPENED_WIDGET_FRAME_BUDGET = 30;
/** Frames to keep the move after it lands: a closing menu or dialog hands focus back to its trigger on the way out. */
const OPENED_WIDGET_SETTLE_FRAMES = 30;
/** Only the latest open moves focus: a control that opens two widgets means the second one. */
let openedWidgetFocusRequest = 0;

/**
 * Moves keyboard focus into the region a control just opened a widget in, which also moves the region highlight there.
 * Leaves focus alone when it is already inside that region, and gives up if the widget never shows.
 */
export const focusOpenedWidget = (region: WidgetRegion, typeId: string): void => {
  // The control that opened the widget, where a closing menu or dialog would put focus back.
  const opener = document.activeElement;
  const request = ++openedWidgetFocusRequest;
  let frames = 0;

  const settle = (container: HTMLElement, remaining: number) => {
    if (request !== openedWidgetFocusRequest) {
      return;
    }

    const active = document.activeElement;

    if (!container.contains(active) && (active === opener || active === document.body || active === null)) {
      container.focus({ preventScroll: true });
    }
    if (remaining > 0) {
      requestAnimationFrame(() => settle(container, remaining - 1));
    }
  };

  const attempt = () => {
    if (request !== openedWidgetFocusRequest) {
      return;
    }

    const container = document.querySelector<HTMLElement>(`[data-focus-region="${region}"]`);
    const shown = container?.querySelector(`[data-hotkey-widget-type-id="${CSS.escape(typeId)}"]`);

    if (container && shown) {
      if (!container.contains(document.activeElement)) {
        if (!container.hasAttribute('tabindex')) {
          container.tabIndex = -1;
        }
        settle(container, OPENED_WIDGET_SETTLE_FRAMES);
      }
      return;
    }

    frames += 1;
    if (frames < OPENED_WIDGET_FRAME_BUDGET) {
      requestAnimationFrame(attempt);
    }
  };

  requestAnimationFrame(attempt);
};
