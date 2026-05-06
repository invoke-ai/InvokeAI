import { Box, Flex, Spinner, Text } from '@invoke-ai/ui-library';
import { VirtuosoMasonry } from '@virtuoso.dev/masonry';
import type { RootState } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { getFocusedRegion } from 'common/hooks/focus';
import { getItemIndex } from 'features/gallery/components/getItemIndex';
import { GalleryImage } from 'features/gallery/components/ImageGrid/GalleryImage';
import { GallerySelectionCountTag } from 'features/gallery/components/ImageGrid/GallerySelectionCountTag';
import { useGalleryImageNames } from 'features/gallery/components/use-gallery-image-names';
import { useGalleryStarImageHotkey } from 'features/gallery/hooks/useGalleryStarImageHotkey';
import type { selectGetImageNamesQueryArgs } from 'features/gallery/store/gallerySelectors';
import {
  selectGalleryImageMinimumWidth,
  selectImageToCompare,
  selectLastSelectedItem,
  selectSelection,
} from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { CSSProperties, MutableRefObject, ReactNode, RefObject } from 'react';
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { imagesApi, useGetImageDTOsByNamesMutation } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { useThrottledCallback } from 'use-debounce';

import {
  deleteMasonryBackgroundInFlightImageNames,
  getMasonryInFlightImageNames,
  getMasonryInitialItemCount,
  getMasonryPrefetchImageNames,
  getMasonrySkippedImageNames,
  getMasonryWarmupImageNames,
  getShouldScheduleNextMasonryWarmupBatch,
  getStaticMasonryColumns,
  getUncachedMasonryImageNames,
  setMasonryBackgroundInFlightImageNames,
} from './masonryImageFetching';
import { getMasonryRenderState } from './masonryRenderState';
import { getMasonrySelectedImageScrollDecision, scrollMasonryImageIntoView } from './masonryScrollIntoView';

type ListImageNamesQueryArgs = ReturnType<typeof selectGetImageNamesQueryArgs>;

type MasonryContext = {
  queryArgs: ListImageNamesQueryArgs;
  registerMissingImageName: (imageName: string) => void;
};

type MasonryMountedRange = { endIndex: number; startIndex: number };
type MasonryScrollDirection = 'down' | 'up' | null;
type StaticMasonryImageDimensions = { height: number; width: number };

const MASONRY_ITEM_PADDING_PX = 2;
const MASONRY_FETCH_DELAY_MS = 50;
const MASONRY_PREFETCH_BATCH_SIZE = 384;
const MASONRY_PREFETCH_DELAY_MS = 75;
const MASONRY_INITIAL_ITEM_COUNT_LIMIT = 256;
const MASONRY_INITIAL_ITEMS_PER_COLUMN = 48;
const MASONRY_INITIAL_MIN_ITEM_COUNT = 128;
const MASONRY_STATIC_RENDER_LIMIT = 1024;
const MASONRY_BACKGROUND_INITIAL_DELAY_MS = 250;
const MASONRY_WARMUP_BATCH_SIZE = 128;
const MASONRY_WARMUP_DELAY_MS = 75;
const MASONRY_WARMUP_IMAGE_LIMIT = 3000;

const canHandleMasonryArrowNavigation = (
  activeTab: ReturnType<typeof selectActiveTab>,
  focusedRegion: ReturnType<typeof getFocusedRegion>
) => {
  if (navigationApi.isViewerArrowNavigationMode(activeTab)) {
    return false;
  }

  if (focusedRegion === 'gallery' || focusedRegion === 'viewer') {
    return true;
  }

  return navigationApi.isDockviewPanelActive(activeTab, VIEWER_PANEL_ID);
};

const useMasonryColumnCount = (rootRef: RefObject<HTMLDivElement>) => {
  const galleryImageMinimumWidth = useAppSelector(selectGalleryImageMinimumWidth);
  const [columnCount, setColumnCount] = useState(1);
  const [hasMeasuredColumnCount, setHasMeasuredColumnCount] = useState(false);

  const recalculateColumnCount = useCallback((): boolean => {
    const rootEl = rootRef.current;
    if (!rootEl) {
      return false;
    }
    const width = rootEl.getBoundingClientRect().width;
    if (!width) {
      return false;
    }
    const nextColumnCount = Math.max(
      1,
      Math.floor((width + MASONRY_ITEM_PADDING_PX * 2) / (galleryImageMinimumWidth + MASONRY_ITEM_PADDING_PX * 2))
    );
    setColumnCount(nextColumnCount);
    setHasMeasuredColumnCount(true);
    return true;
  }, [galleryImageMinimumWidth, rootRef]);

  useLayoutEffect(() => {
    let frame = 0;

    const recalculateUntilMeasured = () => {
      const hasMeasured = recalculateColumnCount();
      if (!hasMeasured) {
        frame = requestAnimationFrame(recalculateUntilMeasured);
      }
    };

    recalculateUntilMeasured();
    const rootEl = rootRef.current;
    if (!rootEl || typeof ResizeObserver === 'undefined') {
      return () => cancelAnimationFrame(frame);
    }
    const observer = new ResizeObserver(recalculateColumnCount);
    observer.observe(rootEl);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [recalculateColumnCount, rootRef]);

  return { columnCount, hasMeasuredColumnCount };
};

const useMasonryThumbnailPreloader = () => {
  const preloadedThumbnailUrlsRef = useRef<Set<string>>(new Set());
  const preloadingImagesRef = useRef<Map<string, HTMLImageElement>>(new Map());

  return useCallback((imageDTOs: ImageDTO[]) => {
    for (const imageDTO of imageDTOs) {
      const { thumbnail_url } = imageDTO;
      if (preloadedThumbnailUrlsRef.current.has(thumbnail_url)) {
        continue;
      }
      preloadedThumbnailUrlsRef.current.add(thumbnail_url);
      const image = new Image();
      image.decoding = 'async';
      image.onload = () => {
        preloadingImagesRef.current.delete(thumbnail_url);
      };
      image.onerror = () => {
        preloadingImagesRef.current.delete(thumbnail_url);
      };
      preloadingImagesRef.current.set(thumbnail_url, image);
      image.src = thumbnail_url;
    }
  }, []);
};

const useMountedMasonryImageFetching = (
  enabled: boolean,
  preloadThumbnails: (imageDTOs: ImageDTO[]) => void,
  visibleInFlightImageNamesRef: MutableRefObject<Set<string>>
) => {
  const store = useAppStore();
  const [getImageDTOsByNames] = useGetImageDTOsByNamesMutation();
  const pendingImageNamesRef = useRef<Set<string>>(new Set());
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flushPendingImageNames = useCallback(() => {
    timeoutRef.current = null;

    if (!enabled) {
      pendingImageNamesRef.current.clear();
      return;
    }

    const pendingImageNames = Array.from(pendingImageNamesRef.current);
    pendingImageNamesRef.current.clear();

    if (pendingImageNames.length === 0) {
      return;
    }

    const cachedImageNames = imagesApi.util.selectCachedArgsForQuery(store.getState(), 'getImageDTO');
    const uncachedImageNames = getUncachedMasonryImageNames(
      pendingImageNames,
      cachedImageNames,
      visibleInFlightImageNamesRef.current
    );

    if (uncachedImageNames.length > 0) {
      for (const imageName of uncachedImageNames) {
        visibleInFlightImageNamesRef.current.add(imageName);
      }
      void getImageDTOsByNames({ image_names: uncachedImageNames })
        .unwrap()
        .then(preloadThumbnails)
        .catch(() => {
          // The visible image components retain their existing loading/fallback behavior.
        })
        .finally(() => {
          for (const imageName of uncachedImageNames) {
            visibleInFlightImageNamesRef.current.delete(imageName);
          }
        });
    }
  }, [enabled, getImageDTOsByNames, preloadThumbnails, store, visibleInFlightImageNamesRef]);

  const registerMissingImageName = useCallback(
    (imageName: string) => {
      if (!enabled) {
        return;
      }

      pendingImageNamesRef.current.add(imageName);
      if (timeoutRef.current === null) {
        timeoutRef.current = setTimeout(flushPendingImageNames, MASONRY_FETCH_DELAY_MS);
      }
    },
    [enabled, flushPendingImageNames]
  );

  useEffect(() => {
    return () => {
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return registerMissingImageName;
};

const getMountedMasonryRange = (rootEl: HTMLDivElement): MasonryMountedRange | null => {
  let startIndex = Number.POSITIVE_INFINITY;
  let endIndex = Number.NEGATIVE_INFINITY;

  for (const el of rootEl.querySelectorAll<HTMLElement>('[data-absolute-index]')) {
    const index = Number.parseInt(el.dataset.absoluteIndex ?? '', 10);
    if (!Number.isFinite(index)) {
      continue;
    }
    startIndex = Math.min(startIndex, index);
    endIndex = Math.max(endIndex, index);
  }

  if (startIndex === Number.POSITIVE_INFINITY || endIndex === Number.NEGATIVE_INFINITY) {
    return null;
  }

  return { endIndex, startIndex };
};

const getMasonryScroller = (rootEl: HTMLDivElement): HTMLElement | null => {
  return rootEl.querySelector<HTMLElement>('[data-testid="virtuoso-scroller"]');
};

const useMasonryImagePrefetching = (
  imageNames: string[],
  columnCount: number,
  rootRef: RefObject<HTMLDivElement>,
  enabled: boolean,
  preloadThumbnails: (imageDTOs: ImageDTO[]) => void,
  visibleInFlightImageNamesRef: MutableRefObject<Set<string>>,
  backgroundInFlightImageNamesRef: MutableRefObject<Map<string, number>>,
  backgroundRequestIdRef: MutableRefObject<number>
) => {
  const store = useAppStore();
  const [getImageDTOsByNames] = useGetImageDTOsByNamesMutation();
  const scrollDirectionRef = useRef<MasonryScrollDirection>('down');
  const scrollTopRef = useRef(0);
  const scrollListenerRef = useRef<(() => void) | null>(null);
  const hasInitialPrefetchDelayElapsedRef = useRef(false);

  const prefetchImages = useCallback(() => {
    const rootEl = rootRef.current;
    if (!enabled || !rootEl) {
      return;
    }

    const cachedImageNames = imagesApi.util.selectCachedArgsForQuery(store.getState(), 'getImageDTO');
    const imageNamesToFetch = getMasonryPrefetchImageNames({
      cachedImageNames,
      batchSize: MASONRY_PREFETCH_BATCH_SIZE,
      columnCount,
      imageNames,
      inFlightImageNames: getMasonryInFlightImageNames({
        backgroundInFlightImageNames: backgroundInFlightImageNamesRef.current,
        visibleInFlightImageNames: visibleInFlightImageNamesRef.current,
      }),
      mountedRange: getMountedMasonryRange(rootEl),
      scrollDirection: scrollDirectionRef.current,
    });

    if (imageNamesToFetch.length === 0) {
      return;
    }

    const requestId = ++backgroundRequestIdRef.current;
    setMasonryBackgroundInFlightImageNames({
      backgroundInFlightImageNames: backgroundInFlightImageNamesRef.current,
      imageNames: imageNamesToFetch,
      requestId,
    });

    void getImageDTOsByNames({ image_names: imageNamesToFetch })
      .unwrap()
      .then(preloadThumbnails)
      .catch(() => {
        // The visible image components retain their existing loading/fallback behavior.
      })
      .finally(() => {
        deleteMasonryBackgroundInFlightImageNames({
          backgroundInFlightImageNames: backgroundInFlightImageNamesRef.current,
          imageNames: imageNamesToFetch,
          requestId,
        });
      });
  }, [
    backgroundInFlightImageNamesRef,
    backgroundRequestIdRef,
    columnCount,
    enabled,
    getImageDTOsByNames,
    imageNames,
    preloadThumbnails,
    rootRef,
    store,
    visibleInFlightImageNamesRef,
  ]);

  const throttledPrefetchImages = useThrottledCallback(prefetchImages, MASONRY_PREFETCH_DELAY_MS);

  useEffect(() => {
    if (!enabled) {
      hasInitialPrefetchDelayElapsedRef.current = false;
      return;
    }

    hasInitialPrefetchDelayElapsedRef.current = false;
    const timeout = setTimeout(() => {
      hasInitialPrefetchDelayElapsedRef.current = true;
      throttledPrefetchImages();
    }, MASONRY_BACKGROUND_INITIAL_DELAY_MS);

    return () => {
      clearTimeout(timeout);
      hasInitialPrefetchDelayElapsedRef.current = false;
    };
  }, [columnCount, enabled, imageNames, throttledPrefetchImages]);

  useEffect(() => {
    const rootEl = rootRef.current;
    if (!enabled || !rootEl) {
      return;
    }

    let scroller: HTMLElement | null = null;
    let frame = 0;

    const connectScroller = () => {
      scroller = getMasonryScroller(rootEl);
      if (!scroller) {
        frame = requestAnimationFrame(connectScroller);
        return;
      }
      scrollTopRef.current = scroller.scrollTop;
      const handleScroll = () => {
        const nextScrollTop = scroller?.scrollTop ?? 0;
        scrollDirectionRef.current = nextScrollTop < scrollTopRef.current ? 'up' : 'down';
        scrollTopRef.current = nextScrollTop;
        throttledPrefetchImages();
      };
      scrollListenerRef.current = handleScroll;
      scroller.addEventListener('scroll', handleScroll, { passive: true });
    };

    frame = requestAnimationFrame(connectScroller);

    return () => {
      cancelAnimationFrame(frame);
      if (scroller && scrollListenerRef.current) {
        scroller.removeEventListener('scroll', scrollListenerRef.current);
        scrollListenerRef.current = null;
      }
    };
  }, [enabled, rootRef, throttledPrefetchImages]);

  useEffect(() => {
    const rootEl = rootRef.current;
    if (!enabled || !rootEl) {
      return;
    }

    const mutationObserver = new MutationObserver(() => {
      if (hasInitialPrefetchDelayElapsedRef.current) {
        throttledPrefetchImages();
      }
    });
    mutationObserver.observe(rootEl, { childList: true, subtree: true });

    const resizeObserver =
      typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver(() => {
            if (hasInitialPrefetchDelayElapsedRef.current) {
              throttledPrefetchImages();
            }
          });
    resizeObserver?.observe(rootEl);

    return () => {
      mutationObserver.disconnect();
      resizeObserver?.disconnect();
    };
  }, [enabled, rootRef, throttledPrefetchImages]);
};

const useMasonryImageWarmup = (
  imageNames: string[],
  enabled: boolean,
  preloadThumbnails: (imageDTOs: ImageDTO[]) => void,
  visibleInFlightImageNamesRef: MutableRefObject<Set<string>>,
  backgroundInFlightImageNamesRef: MutableRefObject<Map<string, number>>,
  backgroundRequestIdRef: MutableRefObject<number>
) => {
  const store = useAppStore();
  const [getImageDTOsByNames] = useGetImageDTOsByNamesMutation();
  const warmupImageNamesRef = useRef<Map<string, number>>(new Map());
  const warmupSkippedImageNamesRef = useRef<Set<string>>(new Set());

  const clearWarmupImageNames = useCallback(() => {
    for (const [imageName, requestId] of warmupImageNamesRef.current) {
      deleteMasonryBackgroundInFlightImageNames({
        backgroundInFlightImageNames: backgroundInFlightImageNamesRef.current,
        imageNames: [imageName],
        requestId,
      });
    }
    warmupImageNamesRef.current.clear();
  }, [backgroundInFlightImageNamesRef]);

  useEffect(() => {
    const warmupSkippedImageNames = warmupSkippedImageNamesRef.current;
    if (!enabled || imageNames.length === 0) {
      clearWarmupImageNames();
      warmupSkippedImageNames.clear();
      return;
    }

    let isCancelled = false;
    let timeout: ReturnType<typeof setTimeout> | null = null;
    clearWarmupImageNames();
    warmupSkippedImageNames.clear();

    const fetchNextBatch = () => {
      if (isCancelled) {
        return;
      }

      const cachedImageNames = imagesApi.util.selectCachedArgsForQuery(store.getState(), 'getImageDTO');
      const imageNamesToFetch = getMasonryWarmupImageNames({
        batchSize: MASONRY_WARMUP_BATCH_SIZE,
        cachedImageNames,
        imageNames,
        inFlightImageNames: getMasonryInFlightImageNames({
          backgroundInFlightImageNames: backgroundInFlightImageNamesRef.current,
          visibleInFlightImageNames: visibleInFlightImageNamesRef.current,
        }),
        maxImageCount: MASONRY_WARMUP_IMAGE_LIMIT,
        skippedImageNames: warmupSkippedImageNames,
      });

      if (imageNamesToFetch.length === 0) {
        return;
      }

      const requestId = ++backgroundRequestIdRef.current;
      setMasonryBackgroundInFlightImageNames({
        backgroundInFlightImageNames: backgroundInFlightImageNamesRef.current,
        imageNames: imageNamesToFetch,
        requestId,
      });
      for (const imageName of imageNamesToFetch) {
        warmupImageNamesRef.current.set(imageName, requestId);
      }

      void getImageDTOsByNames({ image_names: imageNamesToFetch })
        .unwrap()
        .then((imageDTOs) => {
          if (isCancelled) {
            return;
          }
          const skippedImageNames = getMasonrySkippedImageNames({
            requestedImageNames: imageNamesToFetch,
            returnedImageNames: imageDTOs.map((imageDTO) => imageDTO.image_name),
          });
          for (const imageName of skippedImageNames) {
            warmupSkippedImageNames.add(imageName);
          }
          preloadThumbnails(imageDTOs);
          if (
            getShouldScheduleNextMasonryWarmupBatch({
              didFetchBatch: true,
              imageNamesToFetchCount: imageNamesToFetch.length,
              isCancelled,
            })
          ) {
            timeout = setTimeout(fetchNextBatch, MASONRY_WARMUP_DELAY_MS);
          }
        })
        .catch(() => {
          // Visible items keep their normal placeholder behavior if warmup fails.
        })
        .finally(() => {
          deleteMasonryBackgroundInFlightImageNames({
            backgroundInFlightImageNames: backgroundInFlightImageNamesRef.current,
            imageNames: imageNamesToFetch,
            requestId,
          });
          for (const imageName of imageNamesToFetch) {
            if (warmupImageNamesRef.current.get(imageName) === requestId) {
              warmupImageNamesRef.current.delete(imageName);
            }
          }
        });
    };

    timeout = setTimeout(fetchNextBatch, MASONRY_BACKGROUND_INITIAL_DELAY_MS);

    return () => {
      isCancelled = true;
      clearWarmupImageNames();
      warmupSkippedImageNames.clear();
      if (timeout !== null) {
        clearTimeout(timeout);
      }
    };
  }, [
    clearWarmupImageNames,
    enabled,
    getImageDTOsByNames,
    imageNames,
    backgroundInFlightImageNamesRef,
    backgroundRequestIdRef,
    preloadThumbnails,
    store,
    visibleInFlightImageNamesRef,
  ]);
};

const MasonryImagePlaceholder = memo(
  ({ aspectRatio = '1/1', imageName }: { aspectRatio?: string; imageName: string }) => (
    <Flex
      data-item-id={imageName}
      aspectRatio={aspectRatio}
      h="auto"
      w="full"
      bg="base.700"
      borderRadius="base"
      opacity={0.45}
    />
  )
);

MasonryImagePlaceholder.displayName = 'MasonryImagePlaceholder';

type MasonryImageAtPositionProps = {
  context: MasonryContext;
  data: string;
  index: number;
};

const MasonryImageAtPosition = memo(({ data: imageName, context }: MasonryImageAtPositionProps) => {
  const { currentData: imageDTO, isUninitialized } = imagesApi.endpoints.getImageDTO.useQueryState(imageName);
  imagesApi.endpoints.getImageDTO.useQuerySubscription(imageName, { skip: isUninitialized });

  useEffect(() => {
    if (!imageDTO) {
      context.registerMissingImageName(imageName);
    }
  }, [context, imageDTO, imageName]);

  if (!imageDTO) {
    return (
      <Box p={`${MASONRY_ITEM_PADDING_PX}px`}>
        <MasonryImagePlaceholder imageName={imageName} />
      </Box>
    );
  }

  return (
    <Box p={`${MASONRY_ITEM_PADDING_PX}px`}>
      <GalleryImage imageDTO={imageDTO} layout="masonry" />
    </Box>
  );
});

MasonryImageAtPosition.displayName = 'MasonryImageAtPosition';

type StaticMasonryImageGridProps = {
  columnCount: number;
  context: MasonryContext;
  imageNames: string[];
};

const StaticMasonryImageGrid = memo(({ columnCount, context, imageNames }: StaticMasonryImageGridProps) => {
  const selectImageDimensionsByName = useMemo(
    () => (state: RootState) => {
      const dimensions = new Map<string, StaticMasonryImageDimensions>();
      for (const imageName of imageNames) {
        const imageDTO = imagesApi.endpoints.getImageDTO.select(imageName)(state).data;
        if (imageDTO) {
          dimensions.set(imageName, { height: imageDTO.height, width: imageDTO.width });
        }
      }
      return dimensions;
    },
    [imageNames]
  );
  const imageDimensionsByName = useAppSelector(selectImageDimensionsByName, areImageDimensionsByNameEqual);
  const columns = useMemo(
    () => getStaticMasonryColumns({ columnCount, imageDimensionsByName, imageNames }),
    [columnCount, imageDimensionsByName, imageNames]
  );

  return (
    <Flex h="full" w="full" overflowY="auto" alignItems="flex-start">
      {columns.map((column, columnIndex) => (
        <Box key={columnIndex} flexGrow={1} flexBasis={0} minW={0}>
          {column.map(({ imageName, index }, indexInColumn) => (
            <Box
              key={imageName}
              data-absolute-index={index}
              data-column-index={columnIndex}
              data-index={indexInColumn}
              style={STATIC_MASONRY_ITEM_STYLE}
            >
              <MasonryImageAtPosition context={context} data={imageName} index={index} />
            </Box>
          ))}
        </Box>
      ))}
    </Flex>
  );
});

StaticMasonryImageGrid.displayName = 'StaticMasonryImageGrid';

const useMasonryKeyboardNavigation = (
  imageNames: string[],
  columnCount: number,
  rootRef: RefObject<HTMLDivElement>
) => {
  const { dispatch, getState } = useAppStore();
  const activeTab = useAppSelector(selectActiveTab);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const focusedRegion = getFocusedRegion();
      if (!canHandleMasonryArrowNavigation(activeTab, focusedRegion)) {
        return;
      }

      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
        return;
      }

      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (imageNames.length === 0 || columnCount === 0) {
        return;
      }

      event.preventDefault();

      const state = getState();
      const imageName = event.altKey
        ? (selectImageToCompare(state) ?? selectLastSelectedItem(state))
        : selectLastSelectedItem(state);
      const currentIndex = getItemIndex(imageName ?? null, imageNames);

      let newIndex = currentIndex;
      switch (event.key) {
        case 'ArrowLeft':
          newIndex = Math.max(0, currentIndex - 1);
          break;
        case 'ArrowRight':
          newIndex = Math.min(imageNames.length - 1, currentIndex + 1);
          break;
        case 'ArrowUp':
          newIndex = Math.max(0, currentIndex - columnCount);
          break;
        case 'ArrowDown':
          newIndex = Math.min(imageNames.length - 1, currentIndex + columnCount);
          break;
      }

      if (newIndex === currentIndex) {
        return;
      }

      const newImageName = imageNames[newIndex];
      if (!newImageName) {
        return;
      }

      if (event.altKey) {
        dispatch(imageToCompareChanged(newImageName));
      } else {
        dispatch(selectionChanged([newImageName]));
      }

      requestAnimationFrame(() => {
        const rootEl = rootRef.current;
        if (!rootEl) {
          return;
        }
        scrollMasonryImageIntoView({
          imageName: newImageName,
          previousIndex: currentIndex,
          rootEl,
          targetIndex: newIndex,
        });
      });
    },
    [activeTab, columnCount, dispatch, getState, imageNames, rootRef]
  );

  useRegisteredHotkeys({
    id: 'galleryNavLeft',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavRight',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavUp',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavDown',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavLeftAlt',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavRightAlt',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavUpAlt',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavDownAlt',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });
};

const useKeepMasonrySelectedImageInView = (imageNames: string[], rootRef: RefObject<HTMLDivElement>) => {
  const selection = useAppSelector(selectSelection);
  const selectedImageScrollStateRef = useRef({
    hasSelectionChangedSinceMount: false,
    initialSelectedImageName: selection.at(-1) ?? null,
  });

  useEffect(() => {
    const targetImageName = selection.at(-1) ?? null;
    const { nextState, shouldScroll } = getMasonrySelectedImageScrollDecision({
      currentSelectedImageName: targetImageName,
      state: selectedImageScrollStateRef.current,
    });
    selectedImageScrollStateRef.current = nextState;

    if (!shouldScroll || !targetImageName) {
      return;
    }

    const targetIndex = imageNames.indexOf(targetImageName);
    if (targetIndex === -1) {
      return;
    }

    requestAnimationFrame(() => {
      const rootEl = rootRef.current;
      if (!rootEl) {
        return;
      }
      scrollMasonryImageIntoView({
        imageName: targetImageName,
        rootEl,
        targetIndex,
      });
    });
  }, [imageNames, rootRef, selection]);
};

type GalleryImageGridMasonryContentProps = {
  imageNames: string[];
  isLoading: boolean;
  queryArgs: ListImageNamesQueryArgs;
};

const GalleryImageGridMasonryContent = memo(
  ({ imageNames, isLoading, queryArgs }: GalleryImageGridMasonryContentProps) => {
    const { t } = useTranslation();
    const rootRef = useRef<HTMLDivElement>(null);
    const visibleInFlightImageNamesRef = useRef<Set<string>>(new Set());
    const backgroundInFlightImageNamesRef = useRef<Map<string, number>>(new Map());
    const backgroundRequestIdRef = useRef(0);
    const { columnCount, hasMeasuredColumnCount } = useMasonryColumnCount(rootRef);
    const shouldRenderStaticMasonry = imageNames.length <= MASONRY_STATIC_RENDER_LIMIT;
    const initialItemCount = getMasonryInitialItemCount({
      columnCount,
      imageCount: imageNames.length,
      initialItemCountLimit: MASONRY_INITIAL_ITEM_COUNT_LIMIT,
      itemsPerColumn: MASONRY_INITIAL_ITEMS_PER_COLUMN,
      minimumInitialItemCount: MASONRY_INITIAL_MIN_ITEM_COUNT,
    });
    const preloadThumbnails = useMasonryThumbnailPreloader();
    const registerMissingImageName = useMountedMasonryImageFetching(
      !isLoading,
      preloadThumbnails,
      visibleInFlightImageNamesRef
    );
    useMasonryImagePrefetching(
      imageNames,
      columnCount,
      rootRef,
      !isLoading && !shouldRenderStaticMasonry,
      preloadThumbnails,
      visibleInFlightImageNamesRef,
      backgroundInFlightImageNamesRef,
      backgroundRequestIdRef
    );
    useMasonryImageWarmup(
      imageNames,
      !isLoading && !shouldRenderStaticMasonry,
      preloadThumbnails,
      visibleInFlightImageNamesRef,
      backgroundInFlightImageNamesRef,
      backgroundRequestIdRef
    );
    useGalleryStarImageHotkey();
    useMasonryKeyboardNavigation(imageNames, columnCount, rootRef);
    useKeepMasonrySelectedImageInView(imageNames, rootRef);

    const context = useMemo<MasonryContext>(
      () => ({ queryArgs, registerMissingImageName }),
      [queryArgs, registerMissingImageName]
    );
    const renderState = getMasonryRenderState({
      hasMeasuredColumnCount,
      imageCount: imageNames.length,
      isLoading,
    });

    let content: ReactNode = null;

    if (renderState === 'loading') {
      content = (
        <Flex w="full" h="full" alignItems="center" justifyContent="center" gap={4}>
          <Spinner size="lg" opacity={0.3} />
          <Text color="base.300">{t('gallery.loadingGallery')}</Text>
        </Flex>
      );
    } else if (renderState === 'empty') {
      content = (
        <Flex w="full" h="full" alignItems="center" justifyContent="center">
          <Text color="base.300">{t('gallery.noImagesFound')}</Text>
        </Flex>
      );
    } else if (renderState === 'ready') {
      content = shouldRenderStaticMasonry ? (
        <StaticMasonryImageGrid columnCount={columnCount} context={context} imageNames={imageNames} />
      ) : (
        <VirtuosoMasonry<string, MasonryContext>
          columnCount={columnCount}
          context={context}
          data={imageNames}
          initialItemCount={initialItemCount}
          ItemContent={MasonryImageAtPosition}
          style={style}
        />
      );
    }

    return (
      <Box ref={rootRef} position="relative" w="full" h="full">
        {content}
        {renderState === 'ready' ? <GallerySelectionCountTag imageNames={imageNames} /> : null}
      </Box>
    );
  }
);

GalleryImageGridMasonryContent.displayName = 'GalleryImageGridMasonryContent';

export const GalleryImageGridMasonry = memo(() => {
  const { queryArgs, imageNames, isLoading } = useGalleryImageNames();
  return <GalleryImageGridMasonryContent imageNames={imageNames} isLoading={isLoading} queryArgs={queryArgs} />;
});

GalleryImageGridMasonry.displayName = 'GalleryImageGridMasonry';

const style = { height: '100%', width: '100%' };
const STATIC_MASONRY_ITEM_STYLE = { overflowAnchor: 'none' } satisfies CSSProperties;

const areImageDimensionsByNameEqual = (
  a: ReadonlyMap<string, StaticMasonryImageDimensions>,
  b: ReadonlyMap<string, StaticMasonryImageDimensions>
) => {
  if (a.size !== b.size) {
    return false;
  }

  for (const [imageName, dimensions] of a) {
    const otherDimensions = b.get(imageName);
    if (
      !otherDimensions ||
      otherDimensions.width !== dimensions.width ||
      otherDimensions.height !== dimensions.height
    ) {
      return false;
    }
  }

  return true;
};
