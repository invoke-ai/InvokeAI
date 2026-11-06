import type { KeyedTransientStore } from '@platform/state/externalStore';

import { use, useCallback, useSyncExternalStore, type Ref } from 'react';

import { GalleryLayoutStoreContext, type GalleryLayoutMode, type GalleryWidgetProps } from './GalleryUiContext';

type GalleryRegion = GalleryWidgetProps['region'];

/** Only shells interpret arrangement; shared slots must work in either placement. */
const isRailRegion = (region: string): boolean => region === 'left' || region === 'right';

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

/** What a placement shows before its gallery has measured, or while none is mounted there. */
const getUnmeasuredGalleryLayout = (region: GalleryRegion): GalleryLayoutMode =>
  isRailRegion(region) ? 'stacked' : 'wide';

const useGalleryLayoutStore = (): KeyedTransientStore<GalleryRegion, GalleryLayoutMode> => {
  const store = use(GalleryLayoutStoreContext);

  if (!store) {
    throw new Error('Gallery layout requires an App-composed GalleryUiProvider.');
  }

  return store;
};

/**
 * The layout the gallery body in `region` renders, for chrome drawn outside it. Subscribes by key rather than through
 * `store.useValue`: the store comes from context, and hooks must be statically known.
 */
export const useGalleryLayoutMode = (region: GalleryRegion): GalleryLayoutMode => {
  const store = useGalleryLayoutStore();
  const subscribe = useCallback((listener: () => void) => store.subscribeKey(region, listener), [region, store]);
  const getSnapshot = () => store.get(region) ?? getUnmeasuredGalleryLayout(region);

  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

/** Measures the body's width and publishes its layout for the placement; the entry goes when the body unmounts. */
export const useGalleryLayout = (
  region: GalleryRegion
): { layout: GalleryLayoutMode; rootRef: Ref<HTMLDivElement> } => {
  const store = useGalleryLayoutStore();
  const layout = useGalleryLayoutMode(region);
  const rootRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (!node) {
        return;
      }

      const observer = new ResizeObserver((entries) => {
        const widthPx = entries[0]?.contentRect.width;

        if (typeof widthPx === 'number' && widthPx > 0) {
          store.set(region, getGalleryLayout({ region, widthPx }));
        }
      });

      observer.observe(node);

      return () => {
        observer.disconnect();
        store.delete(region);
      };
    },
    [region, store]
  );

  return { layout, rootRef };
};
