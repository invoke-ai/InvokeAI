import { useCallback, useLayoutEffect, useRef, useState, type Ref } from 'react';

import type { GalleryWidgetProps } from './GalleryUiContext';

type GalleryRegion = GalleryWidgetProps['region'];

/** Only shells interpret arrangement; shared slots must work in either placement. */
const isRailRegion = (region: string): boolean => region === 'left' || region === 'right';

export type GalleryLayoutMode = 'stacked' | 'wide';

/** Below this the board column's ~240px leaves too little grid to scan. */
export const GALLERY_WIDE_MIN_WIDTH_PX = 560;

/** Keep docked side panels stacked across resizing to avoid layout switches while dragging their edge. */
export const getGalleryLayout = ({
  region,
  widthPx,
}: {
  region: GalleryRegion;
  widthPx: number;
}): GalleryLayoutMode => {
  if (isRailRegion(region)) {
    return 'stacked';
  }

  return widthPx >= GALLERY_WIDE_MIN_WIDTH_PX ? 'wide' : 'stacked';
};

/** State only changes when the layout flips, not on every measured pixel. */
export const useGalleryLayout = (
  region: GalleryRegion
): { layout: GalleryLayoutMode; rootRef: Ref<HTMLDivElement> } => {
  const [layout, setLayout] = useState<GalleryLayoutMode>(() => (isRailRegion(region) ? 'stacked' : 'wide'));
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
