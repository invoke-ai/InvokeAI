import { Box, Flex, forwardRef, Grid, GridItem, Spinner, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { getFocusedRegion, useIsRegionFocused } from 'common/hooks/focus';
import { useRangeBasedImageFetching } from 'features/gallery/hooks/useRangeBasedImageFetching';
import type { selectGetImageNamesQueryArgs } from 'features/gallery/store/gallerySelectors';
import {
  selectGalleryImageMinimumWidth,
  selectImageToCompare,
  selectLastSelectedItem,
  selectSelection,
  selectSelectionCount,
} from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { MutableRefObject } from 'react';
import React, { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import type {
  GridComponents,
  GridComputeItemKey,
  GridItemContent,
  ListRange,
  ScrollSeekConfiguration,
  VirtuosoGridHandle,
} from 'react-virtuoso';
import { VirtuosoGrid } from 'react-virtuoso';
import { imagesApi, useImageDTO, useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';
import { useDebounce } from 'use-debounce';

import { getItemIndex } from './getItemIndex';
import { getItemsPerRow } from './getItemsPerRow';
import { GalleryImage, GalleryImagePlaceholder } from './ImageGrid/GalleryImage';
import { GallerySelectionCountTag } from './ImageGrid/GallerySelectionCountTag';
import { scrollIntoView } from './scrollIntoView';
import { useGalleryImageNames } from './use-gallery-image-names';
import { useScrollableGallery } from './useScrollableGallery';

type ListImageNamesQueryArgs = ReturnType<typeof selectGetImageNamesQueryArgs>;

type GridContext = {
  queryArgs: ListImageNamesQueryArgs;
  imageNames: string[];
};

/**
 * The gallery uses a windowed list to only render the images that are currently visible in the viewport. It starts by
 * loading a list of all image names for the selected board or view settings. react-virtuoso reports on the currently-
 * visible range of images (plus some "overscan"). We then fetch the full image DTOs only for those images, which are
 * cached by RTK Query. As the user scrolls, the visible range changes and we fetch more image DTOs as needed.
 *
 * This affords a nice UX, where the user can scroll to any part of their gallery. The scrollbar size never changes.
 *
 * We used other approaches in the past:
 * - Infinite scroll: Load an initial chunk of images, then load more as the user scrolls to the bottom. The scrollbar
 * continually shrinks as more images are loaded. This is a poor UX, as the user cannot easily scroll to a specific
 * part of their gallery. It's also pretty complicated to implement within RTK Query, though since we switched, RTK
 * Query now supports infinite queries. It might be easier to do this today.
 * - Traditional pagination: Show a fixed number of images per page, with pagination controls. This is a poor UX,
 * as the user cannot easily scroll to a specific part of their gallery. Gallerys are often very large, and the page
 * size changes depending on the viewport size.
 */

/**
 * Wraps an image - either the placeholder as it is being loaded or the loaded image
 */
const ImageAtPosition = memo(({ imageName }: { index: number; imageName: string }) => {
  /*
   * We rely on the useRangeBasedImageFetching to fetch all image DTOs, caching them with RTK Query.
   *
   * In this component, we just want to consume that cache. Unforutnately, RTK Query does not provide a way to
   * subscribe to a query without triggering a new fetch.
   *
   * There is a hack, though:
   * - https://github.com/reduxjs/redux-toolkit/discussions/4213
   *
   * This essentially means "subscribe to the query once it has some data".
   *
   * One issue with this approach. When an item DTO is already cached - for example, because it is selected and
   * rendered in the viewer - it will show up in the grid before the other items have loaded. This is most
   * noticeable when first loading a board. The first item in the board is selected and rendered immediately in
   * the viewer, caching the DTO. The gallery grid renders, and that first item displays as a thumbnail while the
   * others are still placeholders. After a moment, the rest of the items load up and display as thumbnails.
   */

  // Use `currentData` instead of `data` to prevent a flash of previous image rendered at this index
  const { currentData: imageDTO, isUninitialized } = imagesApi.endpoints.getImageDTO.useQueryState(imageName);
  imagesApi.endpoints.getImageDTO.useQuerySubscription(imageName, { skip: isUninitialized });

  if (!imageDTO) {
    return <GalleryImagePlaceholder data-item-id={imageName} />;
  }

  return <GalleryImage imageDTO={imageDTO} />;
});
ImageAtPosition.displayName = 'ImageAtPosition';

const computeItemKey: GridComputeItemKey<string, GridContext> = (index, imageName, { queryArgs }) => {
  return `${JSON.stringify(queryArgs)}-${imageName ?? index}`;
};

/**
 * Handles keyboard navigation for the gallery.
 */
const useKeyboardNavigation = (
  imageNames: string[],
  virtuosoRef: React.RefObject<VirtuosoGridHandle>,
  rootRef: React.RefObject<HTMLDivElement>
) => {
  const { dispatch, getState } = useAppStore();

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (getFocusedRegion() !== 'gallery') {
        // Only handle keyboard navigation when the gallery is focused
        return;
      }
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

      const imagesPerRow = getItemsPerRow(rootEl);

      if (imagesPerRow === 0) {
        // This can happen if the grid is not yet rendered or has no items
        return;
      }

      event.preventDefault();

      const state = getState();
      const imageName = event.altKey
        ? // When the user holds alt, we are changing the image to compare - if no image to compare is currently selected,
          // we start from the last selected image
          (selectImageToCompare(state) ?? selectLastSelectedItem(state))
        : selectLastSelectedItem(state);

      const currentIndex = getItemIndex(imageName ?? null, imageNames);

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

/**
 * Keeps the last selected image in view when the gallery is scrolled.
 * This is useful for keyboard navigation and ensuring the user can see their selection.
 * It only tracks the last selected image, not the image to compare.
 */
const useKeepSelectedImageInView = (
  imageNames: string[],
  virtuosoRef: React.RefObject<VirtuosoGridHandle>,
  rootRef: React.RefObject<HTMLDivElement>,
  rangeRef: MutableRefObject<ListRange>
) => {
  const selection = useAppSelector(selectSelection);

  useEffect(() => {
    const targetImageName = selection.at(-1);
    const virtuosoGridHandle = virtuosoRef.current;
    const rootEl = rootRef.current;
    const range = rangeRef.current;

    if (!virtuosoGridHandle || !rootEl || !targetImageName || !imageNames || imageNames.length === 0) {
      return;
    }

    setTimeout(() => {
      scrollIntoView(targetImageName, imageNames, rootEl, virtuosoGridHandle, range);
    }, 0);
  }, [imageNames, rangeRef, rootRef, virtuosoRef, selection]);
};

const useStarImageHotkey = () => {
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const selectionCount = useAppSelector(selectSelectionCount);
  const isGalleryFocused = useIsRegionFocused('gallery');
  const imageDTO = useImageDTO(lastSelectedItem);
  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const handleStarHotkey = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!isGalleryFocused) {
      return;
    }
    if (imageDTO.starred) {
      unstarImages({ image_names: [imageDTO.image_name] });
    } else {
      starImages({ image_names: [imageDTO.image_name] });
    }
  }, [imageDTO, isGalleryFocused, starImages, unstarImages]);

  useRegisteredHotkeys({
    id: 'starImage',
    category: 'gallery',
    callback: handleStarHotkey,
    options: { enabled: !!imageDTO && selectionCount === 1 && isGalleryFocused },
    dependencies: [imageDTO, selectionCount, isGalleryFocused, handleStarHotkey],
  });
};

export const GalleryImageGrid = memo(() => {
  const virtuosoRef = useRef<VirtuosoGridHandle>(null);
  const rangeRef = useRef<ListRange>({ startIndex: 0, endIndex: 0 });
  const rootRef = useRef<HTMLDivElement>(null);

  // Get the ordered list of image names - this is our primary data source for virtualization
  const { queryArgs, imageNames, isLoading } = useGalleryImageNames();

  // Use range-based fetching for bulk loading image DTOs into cache based on the visible range
  const { onRangeChanged } = useRangeBasedImageFetching({
    imageNames,
    enabled: !isLoading,
  });

  useStarImageHotkey();
  useKeepSelectedImageInView(imageNames, virtuosoRef, rootRef, rangeRef);
  useKeyboardNavigation(imageNames, virtuosoRef, rootRef);
  const scrollerRef = useScrollableGallery(rootRef);

  /*
   * We have to keep track of the visible range for keep-selected-image-in-view functionality and push the range to
   * the range-based image fetching hook.
   */
  const handleRangeChanged = useCallback(
    (range: ListRange) => {
      rangeRef.current = range;
      onRangeChanged(range);
    },
    [onRangeChanged]
  );

  const context = useMemo<GridContext>(() => ({ imageNames, queryArgs }), [imageNames, queryArgs]);

  if (isLoading) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center" gap={4}>
        <Spinner size="lg" opacity={0.3} />
        <Text color="base.300">Loading gallery...</Text>
      </Flex>
    );
  }

  if (imageNames.length === 0) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center">
        <Text color="base.300">No images found</Text>
      </Flex>
    );
  }

  return (
    // This wrapper component is necessary to initialize the overlay scrollbars!
    <Box data-overlayscrollbars-initialize="" ref={rootRef} position="relative" w="full" h="full">
      <VirtuosoGrid<string, GridContext>
        ref={virtuosoRef}
        context={context}
        data={imageNames}
        increaseViewportBy={4096}
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

GalleryImageGrid.displayName = 'GalleryImageGrid';

const scrollSeekConfiguration: ScrollSeekConfiguration = {
  enter: (velocity) => {
    return Math.abs(velocity) > 2048;
  },
  exit: (velocity) => {
    return velocity === 0;
  },
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
  const [gridTemplateColumns] = useDebounce(_gridTemplateColumns, 300);

  return <Grid ref={ref} gridTemplateColumns={gridTemplateColumns} gap={1} {...rest} />;
});
ListComponent.displayName = 'ListComponent';

const itemContent: GridItemContent<string, GridContext> = (index, imageName) => {
  return <ImageAtPosition index={index} imageName={imageName} />;
};

const ItemComponent: GridComponents<GridContext>['Item'] = forwardRef(({ context: _, ...rest }, ref) => (
  <GridItem ref={ref} aspectRatio="1/1" {...rest} />
));
ItemComponent.displayName = 'ItemComponent';

const ScrollSeekPlaceholderComponent: GridComponents<GridContext>['ScrollSeekPlaceholder'] = (props) => (
  <GridItem aspectRatio="1/1" {...props}>
    <GalleryImagePlaceholder />
  </GridItem>
);

ScrollSeekPlaceholderComponent.displayName = 'ScrollSeekPlaceholderComponent';

const components: GridComponents<GridContext> = {
  Item: ItemComponent,
  List: ListComponent,
  ScrollSeekPlaceholder: ScrollSeekPlaceholderComponent,
};
