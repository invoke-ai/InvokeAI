import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { offsetChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { throttle } from 'lodash-es';
import { useCallback, useEffect, useMemo } from 'react';
import { useListImagesQuery } from 'services/api/endpoints/images';

// Some logic copied from https://github.com/chakra-ui/zag/blob/1925b7342dc76fb06a7ec59a5a4c0063a4620422/packages/machines/pagination/src/pagination.utils.ts

const range = (start: number, end: number) => {
  const length = end - start + 1;
  return Array.from({ length }, (_, idx) => idx + start);
};

export const ELLIPSIS = 'ellipsis' as const;

const getRange = (currentPage: number, totalPages: number, siblingCount: number) => {
  /**
   * `2 * ctx.siblingCount + 5` explanation:
   * 2 * ctx.siblingCount for left/right siblings
   * 5 for 2x left/right ellipsis, 2x first/last page + 1x current page
   *
   * For some page counts (e.g. totalPages: 8, siblingCount: 2),
   * calculated max page is higher than total pages,
   * so we need to take the minimum of both.
   */
  const totalPageNumbers = Math.min(2 * siblingCount + 5, totalPages);

  const firstPageIndex = 1;
  const lastPageIndex = totalPages;

  const leftSiblingIndex = Math.max(currentPage - siblingCount, firstPageIndex);
  const rightSiblingIndex = Math.min(currentPage + siblingCount, lastPageIndex);

  const showLeftEllipsis = leftSiblingIndex > firstPageIndex + 1;
  const showRightEllipsis = rightSiblingIndex < lastPageIndex - 1;

  const itemCount = totalPageNumbers - 2; // 2 stands for one ellipsis and either first or last page

  if (!showLeftEllipsis && showRightEllipsis) {
    const leftRange = range(1, itemCount);
    return [...leftRange, ELLIPSIS, lastPageIndex];
  }

  if (showLeftEllipsis && !showRightEllipsis) {
    const rightRange = range(lastPageIndex - itemCount + 1, lastPageIndex);
    return [firstPageIndex, ELLIPSIS, ...rightRange];
  }

  if (showLeftEllipsis && showRightEllipsis) {
    const middleRange = range(leftSiblingIndex, rightSiblingIndex);
    return [firstPageIndex, ELLIPSIS, ...middleRange, ELLIPSIS, lastPageIndex];
  }

  const fullRange = range(firstPageIndex, lastPageIndex);
  return fullRange as (number | 'ellipsis')[];
};

const selectOffset = createSelector(selectGallerySlice, (gallery) => gallery.offset);
const selectLimit = createSelector(selectGallerySlice, (gallery) => gallery.limit);

export const useGalleryPagination = () => {
  const dispatch = useAppDispatch();
  const offset = useAppSelector(selectOffset);
  const limit = useAppSelector(selectLimit);
  const queryArgs = useAppSelector(selectListImagesQueryArgs);

  const { count, total } = useListImagesQuery(queryArgs, {
    selectFromResult: ({ data }) => ({ count: data?.items.length ?? 0, total: data?.total ?? 0 }),
  });

  const currentPage = useMemo(() => Math.ceil(offset / (limit || 0)), [offset, limit]);
  const pages = useMemo(() => Math.ceil(total / (limit || 0)), [total, limit]);

  const isNextEnabled = useMemo(() => {
    if (!count) {
      return false;
    }
    return currentPage + 1 < pages;
  }, [count, currentPage, pages]);
  const isPrevEnabled = useMemo(() => {
    if (!count) {
      return false;
    }
    return offset > 0;
  }, [count, offset]);

  const onOffsetChanged = useCallback(
    (arg: Parameters<typeof offsetChanged>[0]) => {
      dispatch(offsetChanged(arg));
    },
    [dispatch]
  );

  const throttledOnOffsetChanged = useMemo(() => throttle(onOffsetChanged, 500), [onOffsetChanged]);

  const goNext = useCallback(
    (withHotkey?: 'arrow' | 'alt+arrow') => {
      throttledOnOffsetChanged({ offset: offset + (limit || 0), withHotkey });
    },
    [throttledOnOffsetChanged, offset, limit]
  );

  const goPrev = useCallback(
    (withHotkey?: 'arrow' | 'alt+arrow') => {
      throttledOnOffsetChanged({ offset: Math.max(offset - (limit || 0), 0), withHotkey });
    },
    [throttledOnOffsetChanged, offset, limit]
  );

  const goToPage = useCallback(
    (page: number) => {
      throttledOnOffsetChanged({ offset: page * (limit || 0) });
    },
    [throttledOnOffsetChanged, limit]
  );
  const goToFirst = useCallback(() => {
    throttledOnOffsetChanged({ offset: 0 });
  }, [throttledOnOffsetChanged]);
  const goToLast = useCallback(() => {
    throttledOnOffsetChanged({ offset: (pages - 1) * (limit || 0) });
  }, [throttledOnOffsetChanged, pages, limit]);

  // handle when total/pages decrease and user is on high page number (ie bulk removing or deleting)
  useEffect(() => {
    if (pages && currentPage + 1 > pages) {
      goToLast();
    }
  }, [currentPage, pages, goToLast]);

  const pageButtons = useMemo(() => {
    if (pages > 7) {
      return getRange(currentPage + 1, pages, 1);
    }
    return range(1, pages);
  }, [currentPage, pages]);
  const isFirstEnabled = useMemo(() => currentPage > 0, [currentPage]);
  const isLastEnabled = useMemo(() => currentPage < pages - 1, [currentPage, pages]);

  const rangeDisplay = useMemo(() => {
    const startItem = currentPage * (limit || 0) + 1;
    const endItem = Math.min((currentPage + 1) * (limit || 0), total);
    return `${startItem}-${endItem} of ${total}`;
  }, [total, currentPage, limit]);

  const numberOnPage = useMemo(() => {
    return Math.min((currentPage + 1) * (limit || 0), total);
  }, [currentPage, limit, total]);

  return {
    count,
    total,
    currentPage,
    pages,
    isNextEnabled,
    isPrevEnabled,
    goNext,
    goPrev,
    goToPage,
    goToFirst,
    goToLast,
    pageButtons,
    isFirstEnabled,
    isLastEnabled,
    rangeDisplay,
    numberOnPage,
  };
};
