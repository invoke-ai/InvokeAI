import { Box, Flex, forwardRef, Grid, GridItem, Skeleton, Spinner, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import type { selectListImageNamesQueryArgs } from 'features/gallery/store/gallerySelectors';
import {
  LIMIT,
  selectGalleryImageMinimumWidth,
  selectImageToCompare,
  selectLastSelectedImage,
} from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import type { MutableRefObject, RefObject } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  GridComponents,
  GridComputeItemKey,
  GridItemContent,
  ListRange,
  ScrollSeekConfiguration,
  VirtuosoGridHandle,
} from 'react-virtuoso';
import { VirtuosoGrid } from 'react-virtuoso';
import { useListImagesQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { useDebounce } from 'use-debounce';

import { GalleryImage } from './ImageGrid/GalleryImage';
import { GallerySelectionCountTag } from './ImageGrid/GallerySelectionCountTag';
import { useGalleryImageNames } from './use-gallery-image-names';

const log = logger('gallery');

// Constants
const VIEWPORT_BUFFER = 2048;
const SCROLL_SEEK_VELOCITY_THRESHOLD = 4096;
const DEBOUNCE_DELAY = 500;
const SPINNER_OPACITY = 0.3;

type ListImageNamesQueryArgs = ReturnType<typeof selectListImageNamesQueryArgs>;

type GridContext = {
  queryArgs: ListImageNamesQueryArgs;
  imageNames: string[];
};

// Hook to get an image DTO from cache or trigger loading
const useImageDTOFromListQuery = (
  index: number,
  imageName: string,
  queryArgs: ListImageNamesQueryArgs
): ImageDTO | null => {
  const { arg, options } = useMemo(() => {
    const pageOffset = Math.floor(index / LIMIT) * LIMIT;
    return {
      arg: {
        ...queryArgs,
        offset: pageOffset,
        limit: LIMIT,
      } satisfies Parameters<typeof useListImagesQuery>[0],
      options: {
        selectFromResult: ({ data }) => {
          const imageDTO = data?.items?.[index - pageOffset] || null;
          if (imageDTO && imageDTO.image_name !== imageName) {
            log.warn(`Image at index ${index} does not match expected image name ${imageName}`);
            return { imageDTO: null };
          }
          return { imageDTO };
        },
      } satisfies Parameters<typeof useListImagesQuery>[1],
    };
  }, [index, queryArgs, imageName]);

  const { imageDTO } = useListImagesQuery(arg, options);

  return imageDTO;
};

// Individual image component that gets its data from RTK Query cache
const ImageAtPosition = memo(
  ({ index, queryArgs, imageName }: { index: number; imageName: string; queryArgs: ListImageNamesQueryArgs }) => {
    const imageDTO = useImageDTOFromListQuery(index, imageName, queryArgs);

    if (!imageDTO) {
      return <Skeleton w="full" h="full" />;
    }

    return <GalleryImage imageDTO={imageDTO} />;
  }
);
ImageAtPosition.displayName = 'ImageAtPosition';

// Memoized compute key function using image names
const computeItemKey: GridComputeItemKey<string, GridContext> = (_index, imageName, { queryArgs }) => {
  return `${JSON.stringify(queryArgs)}-${imageName}`;
};

// Physical DOM-based grid calculation using refs (based on working old implementation)
const getImagesPerRow = (rootEl: HTMLDivElement): number => {
  // Start from root and find virtuoso grid elements
  const gridElement = rootEl.querySelector('.virtuoso-grid-list');

  if (!gridElement) {
    return 0;
  }

  const firstGridItem = gridElement.querySelector('.virtuoso-grid-item');

  if (!firstGridItem) {
    return 0;
  }

  const itemRect = firstGridItem.getBoundingClientRect();
  const containerRect = gridElement.getBoundingClientRect();

  // Get the computed gap from CSS
  const gridStyle = window.getComputedStyle(gridElement);
  const gapValue = gridStyle.gap;
  const gap = parseFloat(gapValue);

  if (isNaN(gap) || !itemRect.width || !itemRect.height || !containerRect.width || !containerRect.height) {
    return 0;
  }

  // Use the exact calculation from the working old implementation
  let imagesPerRow = 0;
  let spaceUsed = 0;

  // Floating point precision can cause imagesPerRow to be 1 too small. Adding 1px to the container size fixes
  // this, without the possibility of accidentally adding an extra column.
  while (spaceUsed + itemRect.width <= containerRect.width + 1) {
    imagesPerRow++; // Increment the number of images
    spaceUsed += itemRect.width; // Add image size to the used space
    if (spaceUsed + gap <= containerRect.width) {
      spaceUsed += gap; // Add gap size to the used space after each image except after the last image
    }
  }

  return Math.max(1, imagesPerRow);
};

// Check if an item at a given index is visible in the viewport
const scrollIntoView = (
  index: number,
  rootEl: HTMLDivElement,
  virtuosoGridHandle: VirtuosoGridHandle,
  range: ListRange
) => {
  if (range.endIndex === 0) {
    return;
  }

  // First get the virtuoso grid list root element
  const gridList = rootEl.querySelector('.virtuoso-grid-list') as HTMLElement;

  if (!gridList) {
    // No grid - cannot scroll!
    return;
  }

  // Then find the specific item within the grid list
  const targetItem = gridList.querySelector(`.virtuoso-grid-item[data-index="${index}"]`) as HTMLElement;

  if (!targetItem) {
    if (index > range.endIndex) {
      virtuosoGridHandle.scrollToIndex({
        index,
        behavior: 'auto',
        align: 'start',
      });
    } else if (index < range.startIndex) {
      virtuosoGridHandle.scrollToIndex({
        index,
        behavior: 'auto',
        align: 'end',
      });
    } else {
      log.warn(`Unable to find item index ${index} but it is in range ${range.startIndex}-${range.endIndex}`);
    }
    return;
  }

  const itemRect = targetItem.getBoundingClientRect();
  const rootRect = rootEl.getBoundingClientRect();

  if (itemRect.top < rootRect.top) {
    virtuosoGridHandle.scrollToIndex({
      index,
      behavior: 'auto',
      align: 'start',
    });
  } else if (itemRect.bottom > rootRect.bottom) {
    virtuosoGridHandle.scrollToIndex({
      index,
      behavior: 'auto',
      align: 'end',
    });
  }

  return;
};

const getImageIndex = (imageName: string | undefined | null, imageNames: string[]) => {
  if (!imageName || imageNames.length === 0) {
    return 0;
  }
  const index = imageNames.findIndex((n) => n === imageName);
  return index >= 0 ? index : 0;
};

// Hook for keyboard navigation using physical DOM measurements
const useKeyboardNavigation = (
  imageNames: string[],
  virtuosoRef: React.RefObject<VirtuosoGridHandle>,
  rootRef: React.RefObject<HTMLDivElement>
) => {
  const { dispatch, getState } = useAppStore();

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Only handle arrow keys
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
        return;
      }
      // Don't interfere if user is typing in an input
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      const rootEl = rootRef.current;
      const virtuosoGridHandle = virtuosoRef.current;

      if (!rootEl || !virtuosoGridHandle) {
        return;
      }

      if (imageNames.length === 0) {
        return;
      }

      const imagesPerRow = getImagesPerRow(rootEl);

      if (imagesPerRow === 0) {
        // This can happen if the grid is not yet rendered or has no items
        return;
      }

      event.preventDefault();

      const imageName = event.altKey
        ? // When the user holds alt, we are changing the image to compare - if no image to compare is currently selected,
          // we start from the last selected image
          (selectImageToCompare(getState()) ?? selectLastSelectedImage(getState()))
        : selectLastSelectedImage(getState());

      const currentIndex = getImageIndex(imageName, imageNames);

      let newIndex = currentIndex;

      switch (event.key) {
        case 'ArrowLeft':
          if (currentIndex > 0) {
            newIndex = currentIndex - 1;
            // } else {
            //   // Wrap to last image
            //   newIndex = imageNames.length - 1;
          }
          break;
        case 'ArrowRight':
          if (currentIndex < imageNames.length - 1) {
            newIndex = currentIndex + 1;
            // } else {
            //   // Wrap to first image
            //   newIndex = 0;
          }
          break;
        case 'ArrowUp':
          // If on first row, stay on current image
          if (currentIndex < imagesPerRow) {
            newIndex = currentIndex;
          } else {
            newIndex = Math.max(0, currentIndex - imagesPerRow);
          }
          break;
        case 'ArrowDown':
          // If no images below, stay on current image
          if (currentIndex >= imageNames.length - imagesPerRow) {
            newIndex = currentIndex;
          } else {
            newIndex = Math.min(imageNames.length - 1, currentIndex + imagesPerRow);
          }
          break;
      }

      if (newIndex !== currentIndex && newIndex >= 0 && newIndex < imageNames.length) {
        const newImageName = imageNames[newIndex];
        if (newImageName) {
          if (event.altKey) {
            dispatch(imageToCompareChanged(newImageName));
          } else {
            dispatch(selectionChanged([newImageName]));
          }
        }
      }
    },
    [rootRef, virtuosoRef, imageNames, getState, dispatch]
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

const useKeepSelectedImageInView = (
  imageNames: string[],
  virtuosoRef: React.RefObject<VirtuosoGridHandle>,
  rootRef: React.RefObject<HTMLDivElement>,
  rangeRef: MutableRefObject<ListRange>
) => {
  const imageName = useAppSelector(selectLastSelectedImage);

  useEffect(() => {
    const virtuosoGridHandle = virtuosoRef.current;
    const rootEl = rootRef.current;
    const range = rangeRef.current;

    if (!virtuosoGridHandle || !rootEl || !imageNames || imageNames.length === 0) {
      return;
    }
    const index = imageName ? imageNames.indexOf(imageName) : 0;
    if (index === -1) {
      return;
    }
    scrollIntoView(index, rootEl, virtuosoGridHandle, range);
  }, [imageName, imageNames, rangeRef, rootRef, virtuosoRef]);
};

const useScrollableGallery = (rootRef: RefObject<HTMLDivElement>) => {
  const [scroller, scrollerRef] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars({
    defer: true,
    events: {
      initialized(osInstance) {
        // force overflow styles
        const { viewport } = osInstance.elements();
        viewport.style.overflowX = `var(--os-viewport-overflow-x)`;
        viewport.style.overflowY = `var(--os-viewport-overflow-y)`;
      },
    },
  });

  useEffect(() => {
    const { current: root } = rootRef;

    if (scroller && root) {
      initialize({
        target: root,
        elements: {
          viewport: scroller,
        },
      });
    }

    return () => {
      osInstance()?.destroy();
    };
  }, [scroller, initialize, osInstance, rootRef]);

  return scrollerRef;
};

// Main gallery component
export const NewGallery = memo(() => {
  const virtuosoRef = useRef<VirtuosoGridHandle>(null);
  const rangeRef = useRef<ListRange>({ startIndex: 0, endIndex: 0 });
  const rootRef = useRef<HTMLDivElement>(null);

  // Get the ordered list of image names - this is our primary data source for virtualization
  const { queryArgs, imageNames, isLoading } = useGalleryImageNames();

  useKeepSelectedImageInView(imageNames, virtuosoRef, rootRef, rangeRef);
  useKeyboardNavigation(imageNames, virtuosoRef, rootRef);
  const scrollerRef = useScrollableGallery(rootRef);

  // We have to keep track of the visible range for keep-selected-image-in-view functionality
  const handleRangeChanged = useCallback((range: ListRange) => {
    rangeRef.current = range;
  }, []);

  const context = useMemo(
    () =>
      ({
        imageNames,
        queryArgs,
      }) satisfies GridContext,
    [imageNames, queryArgs]
  );

  // Item content function
  const itemContent: GridItemContent<string, GridContext> = useCallback((index, imageName, ctx) => {
    return <ImageAtPosition index={index} imageName={imageName} queryArgs={ctx.queryArgs} />;
  }, []);

  if (isLoading) {
    return (
      <Flex height="100%" alignItems="center" justifyContent="center">
        <Spinner size="lg" opacity={SPINNER_OPACITY} />
        <Text ml={4}>Loading gallery...</Text>
      </Flex>
    );
  }

  if (imageNames.length === 0) {
    return (
      <Flex height="100%" alignItems="center" justifyContent="center">
        <Text color="base.300">No images found</Text>
      </Flex>
    );
  }

  return (
    <Box data-overlayscrollbars-initialize="" ref={rootRef} position="relative" w="full" h="full">
      <VirtuosoGrid<string, GridContext>
        ref={virtuosoRef}
        context={context}
        data={imageNames}
        increaseViewportBy={VIEWPORT_BUFFER}
        itemContent={itemContent}
        computeItemKey={computeItemKey}
        components={components}
        style={style}
        scrollerRef={scrollerRef}
        scrollSeekConfiguration={scrollSeekConfiguration}
        rangeChanged={handleRangeChanged}
      />
      <GallerySelectionCountTag />
    </Box>
  );
});

NewGallery.displayName = 'NewGallery';

const scrollSeekConfiguration: ScrollSeekConfiguration = {
  enter: (velocity) => velocity > SCROLL_SEEK_VELOCITY_THRESHOLD,
  exit: (velocity) => velocity === 0,
};

// Styles
const style = { height: '100%', width: '100%' };

const selectGridTemplateColumns = createSelector(
  selectGalleryImageMinimumWidth,
  (galleryImageMinimumWidth) => `repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr))`
);

// Grid components
const ListComponent: GridComponents<GridContext>['List'] = forwardRef(({ context: _, ...rest }, ref) => {
  const _gridTemplateColumns = useAppSelector(selectGridTemplateColumns);
  const [gridTemplateColumns] = useDebounce(_gridTemplateColumns, DEBOUNCE_DELAY);

  return <Grid ref={ref} gridTemplateColumns={gridTemplateColumns} gap={1} {...rest} />;
});
ListComponent.displayName = 'ListComponent';

const ItemComponent: GridComponents<GridContext>['Item'] = forwardRef(({ context: _, ...rest }, ref) => (
  <GridItem ref={ref} aspectRatio="1/1" {...rest} />
));
ItemComponent.displayName = 'ItemComponent';

const ScrollSeekPlaceholderComponent: GridComponents<GridContext>['ScrollSeekPlaceholder'] = forwardRef(
  (props, ref) => (
    <GridItem ref={ref} aspectRatio="1/1" {...props}>
      <Skeleton w="full" h="full" />
    </GridItem>
  )
);
ScrollSeekPlaceholderComponent.displayName = 'ScrollSeekPlaceholderComponent';

const components: GridComponents<GridContext> = {
  Item: ItemComponent,
  List: ListComponent,
  ScrollSeekPlaceholder: ScrollSeekPlaceholderComponent,
};
