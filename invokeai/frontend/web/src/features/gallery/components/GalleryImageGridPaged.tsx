import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { GalleryImageGridContent } from 'features/gallery/components/GalleryImageGrid';
import { GalleryPaginationPaged } from 'features/gallery/components/ImageGrid/GalleryPaginationPaged';
import { useGalleryImageNames } from 'features/gallery/components/use-gallery-image-names';
import { selectGalleryImageMinimumWidth, selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';

import { getItemsPerPage } from './getItemsPerPage';

const FALLBACK_PAGE_SIZE = 200;

export const GalleryImageGridPaged = memo(() => {
  const { queryArgs, imageNames, isLoading } = useGalleryImageNames();
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const galleryImageMinimumWidth = useAppSelector(selectGalleryImageMinimumWidth);
  const [pageIndex, setPageIndex] = useState(0);
  const [pageSize, setPageSize] = useState(FALLBACK_PAGE_SIZE);
  const gridRootRef = useRef<HTMLDivElement>(null);
  const lastSelectedRef = useRef<string | null>(null);

  const pageCount = Math.ceil(imageNames.length / pageSize);
  const pageImageNames = useMemo(() => {
    const start = pageIndex * pageSize;
    return imageNames.slice(start, start + pageSize);
  }, [imageNames, pageIndex, pageSize]);

  useEffect(() => {
    if (pageIndex >= pageCount && pageCount > 0) {
      setPageIndex(pageCount - 1);
    }
  }, [pageCount, pageIndex]);

  useEffect(() => {
    if (!lastSelectedItem) {
      lastSelectedRef.current = null;
      return;
    }
    if (lastSelectedRef.current === lastSelectedItem) {
      return;
    }
    lastSelectedRef.current = lastSelectedItem;
    const selectedIndex = imageNames.indexOf(lastSelectedItem);
    if (selectedIndex === -1) {
      return;
    }
    const nextPageIndex = Math.floor(selectedIndex / pageSize);
    if (nextPageIndex !== pageIndex) {
      setPageIndex(nextPageIndex);
    }
  }, [imageNames, lastSelectedItem, pageIndex, pageSize]);

  const recalculatePageSize = useCallback(() => {
    const rootEl = gridRootRef.current;
    if (!rootEl) {
      return;
    }
    const nextPageSize = getItemsPerPage(rootEl);
    if (nextPageSize > 0 && nextPageSize !== pageSize) {
      setPageSize(nextPageSize);
    }
  }, [pageSize]);

  useLayoutEffect(() => {
    if (isLoading) {
      return;
    }
    let frame = 0;
    let attempts = 0;
    const tick = () => {
      const rootEl = gridRootRef.current;
      if (!rootEl) {
        frame = requestAnimationFrame(tick);
        return;
      }
      const nextPageSize = getItemsPerPage(rootEl);
      if (nextPageSize > 0) {
        if (nextPageSize !== pageSize) {
          setPageSize(nextPageSize);
        }
        return;
      }
      if (attempts < 10) {
        attempts += 1;
        frame = requestAnimationFrame(tick);
      }
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [galleryImageMinimumWidth, imageNames.length, isLoading, pageIndex, pageSize]);

  useEffect(() => {
    if (isLoading) {
      return;
    }
    const timeout = setTimeout(() => {
      recalculatePageSize();
      requestAnimationFrame(recalculatePageSize);
    }, 350);
    return () => clearTimeout(timeout);
  }, [galleryImageMinimumWidth, isLoading, recalculatePageSize]);

  useEffect(() => {
    const rootEl = gridRootRef.current;
    if (!rootEl || typeof ResizeObserver === 'undefined') {
      return;
    }
    const observer = new ResizeObserver(() => {
      recalculatePageSize();
    });
    observer.observe(rootEl);
    return () => observer.disconnect();
  }, [galleryImageMinimumWidth, recalculatePageSize]);

  useEffect(() => {
    const rootEl = gridRootRef.current;
    if (!rootEl || typeof MutationObserver === 'undefined') {
      return;
    }
    const observer = new MutationObserver(() => {
      recalculatePageSize();
    });
    observer.observe(rootEl, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [recalculatePageSize]);

  const handleTabChange = useCallback((index: number) => {
    setPageIndex(index);
  }, []);

  const handlePreviousPage = useCallback(() => {
    setPageIndex((prev) => Math.max(0, prev - 1));
  }, []);

  const handleNextPage = useCallback(() => {
    setPageIndex((prev) => Math.min(pageCount - 1, prev + 1));
  }, [pageCount]);

  const handlePageInputChange = useCallback(
    (valueAsString: string, valueAsNumber: number) => {
      if (!valueAsString) {
        return;
      }
      if (Number.isNaN(valueAsNumber)) {
        return;
      }
      const nextIndex = Math.min(Math.max(valueAsNumber, 1), pageCount) - 1;
      setPageIndex(nextIndex);
    },
    [pageCount]
  );

  if (isLoading || imageNames.length === 0) {
    return <GalleryImageGridContent imageNames={imageNames} isLoading={isLoading} queryArgs={queryArgs} />;
  }

  return (
    <Flex w="full" h="full" flexDir="column" gap={2}>
      <GalleryPaginationPaged
        pageIndex={pageIndex}
        pageCount={pageCount}
        onPrev={handlePreviousPage}
        onNext={handleNextPage}
        onGoToPage={handleTabChange}
        onPageInputChange={handlePageInputChange}
      />
      <Flex w="full" h="full">
        <GalleryImageGridContent
          imageNames={pageImageNames}
          isLoading={false}
          queryArgs={queryArgs}
          rootRef={gridRootRef}
        />
      </Flex>
    </Flex>
  );
});

GalleryImageGridPaged.displayName = 'GalleryImageGridPaged';
