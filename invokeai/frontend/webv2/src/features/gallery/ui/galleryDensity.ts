import { useCallback, useLayoutEffect, useRef, useState, type Ref } from 'react';

import type { GalleryWidgetProps } from './GalleryUiContext';

type GalleryRegion = GalleryWidgetProps['region'];

/**
 * The gallery's two arrangements of one component set. Only the two shell
 * components read it: every slot renders one way, so a size-dependent
 * difference must be expressible as "which shell puts this where".
 */
export type GalleryLayoutMode = 'stacked' | 'wide';

/** Below this the board column's ~240px leaves too little grid to scan. */
export const GALLERY_WIDE_MIN_WIDTH_PX = 560;

/**
 * Side panels are always stacked, whatever they measure: a docked inspector
 * keeps one arrangement so it does not reflow as the user drags its edge.
 * (Their maximum is 720px, so the threshold below is reachable there — the
 * branch is the reason they never take it, not the width.)
 */
export const getGalleryLayout = ({
  region,
  widthPx,
}: {
  region: GalleryRegion;
  widthPx: number;
}): GalleryLayoutMode => {
  if (region === 'left' || region === 'right') {
    return 'stacked';
  }

  return widthPx >= GALLERY_WIDE_MIN_WIDTH_PX ? 'wide' : 'stacked';
};

/** State only changes when the layout flips, not on every measured pixel. */
export const useGalleryLayout = (
  region: GalleryRegion
): { layout: GalleryLayoutMode; rootRef: Ref<HTMLDivElement> } => {
  const [layout, setLayout] = useState<GalleryLayoutMode>(() =>
    region === 'left' || region === 'right' ? 'stacked' : 'wide'
  );
  const regionRef = useRef(region);

  useLayoutEffect(() => {
    regionRef.current = region;
  }, [region]);

  const applyMeasuredWidth = useCallback((widthPx: number) => {
    setLayout(getGalleryLayout({ region: regionRef.current, widthPx }));
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

  return { layout, rootRef };
};
