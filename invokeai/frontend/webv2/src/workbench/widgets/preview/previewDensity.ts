import type { WorkbenchRegion } from '@workbench/widgetContracts';

import { useCallback, useLayoutEffect, useRef, useState, type Ref } from 'react';

/**
 * Share density across Preview zones: full exposes all controls, compact reduces actions/summaries, and minimal
 * keeps frame, progress, and navigation.
 */
const isRailRegion = (region: string): boolean => region === 'left' || region === 'right';

export type PreviewDensity = 'full' | 'compact' | 'minimal';

export const PREVIEW_MINIMAL_MAX_WIDTH_PX = 280;
export const PREVIEW_FULL_MIN_WIDTH_PX = 560;

export const getPreviewDensity = ({
  region,
  widthPx,
}: {
  region: WorkbenchRegion;
  widthPx: number;
}): PreviewDensity => {
  if (widthPx < PREVIEW_MINIMAL_MAX_WIDTH_PX) {
    return 'minimal';
  }

  if (isRailRegion(region)) {
    return 'compact';
  }

  return widthPx >= PREVIEW_FULL_MIN_WIDTH_PX ? 'full' : 'compact';
};

/** Observe root width with a cleaned-up ref callback; update state only when its density bucket changes. */
export const usePreviewDensity = (
  region: WorkbenchRegion
): { density: PreviewDensity; rootRef: Ref<HTMLDivElement> } => {
  const [density, setDensity] = useState<PreviewDensity>(isRailRegion(region) ? 'compact' : 'full');
  const regionRef = useRef(region);
  useLayoutEffect(() => {
    regionRef.current = region;
  }, [region]);
  const applyMeasuredWidth = useCallback((widthPx: number) => {
    setDensity(getPreviewDensity({ region: regionRef.current, widthPx }));
  }, []);
  const [rootRef] = useState(() => (node: HTMLDivElement | null) => {
    if (!node) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const widthPx = entries[0]?.contentRect.width;

      if (typeof widthPx === 'number' && widthPx > 0) {
        applyMeasuredWidth(widthPx);
      }
    });

    observer.observe(node);

    return () => observer.disconnect();
  });

  return { density, rootRef };
};
