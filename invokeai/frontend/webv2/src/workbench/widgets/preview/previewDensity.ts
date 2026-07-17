import type { WorkbenchRegion } from '@workbench/types';

import { useEffectEvent, useState, type Ref } from 'react';

/**
 * Single source of truth for the preview widget's responsive behavior. Every
 * zone component (frame, filmstrip, footer) branches on this value instead of
 * defining its own breakpoints, so side-panel ergonomics are tuned in one place.
 *
 * - `full`     — center region with room to breathe: full action strip,
 *                filmstrip, metadata rows, footer message line.
 * - `compact`  — side panels and narrow center: star + overflow menu,
 *                one-line summaries, no footer message line.
 * - `minimal`  — very narrow: frame + hairline progress + prev/next only.
 */
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

  if (region === 'left' || region === 'right') {
    return 'compact';
  }

  return widthPx >= PREVIEW_FULL_MIN_WIDTH_PX ? 'full' : 'compact';
};

/**
 * Observes the widget root's width and returns the derived density. The
 * ResizeObserver lives in a ref callback (with cleanup) so there is no effect
 * to keep in sync; state only changes when the density bucket changes.
 */
export const usePreviewDensity = (
  region: WorkbenchRegion
): { density: PreviewDensity; rootRef: Ref<HTMLDivElement> } => {
  const [density, setDensity] = useState<PreviewDensity>(region === 'left' || region === 'right' ? 'compact' : 'full');
  const applyMeasuredWidth = useEffectEvent((widthPx: number) => {
    setDensity(getPreviewDensity({ region, widthPx }));
  });
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
