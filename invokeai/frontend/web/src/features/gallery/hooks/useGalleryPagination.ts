import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { offsetChanged } from 'features/gallery/store/gallerySlice';
import { useCallback, useMemo } from 'react';
import { useListImagesQuery } from 'services/api/endpoints/images';

export const useGalleryPagination = (pageButtonsPerSide: number = 2) => {
  const dispatch = useAppDispatch();
  const { offset, limit } = useAppSelector((s) => s.gallery);
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

  const goNext = useCallback(() => {
    dispatch(offsetChanged(offset + (limit || 0)));
  }, [dispatch, offset, limit]);

  const goPrev = useCallback(() => {
    dispatch(offsetChanged(Math.max(offset - (limit || 0), 0)));
  }, [dispatch, offset, limit]);

  const goToPage = useCallback(
    (page: number) => {
      const p = Math.max(0, Math.min(page, pages - 1));
      dispatch(offsetChanged(page * (limit || 0)));
    },
    [dispatch, pages, limit]
  );
  const goToFirst = useCallback(() => {
    dispatch(offsetChanged(0));
  }, [dispatch]);
  const goToLast = useCallback(() => {
    dispatch(offsetChanged((pages - 1) * (limit || 0)));
  }, [dispatch, pages, limit]);

  // calculate the page buttons to display - current page with 3 around it
  const pageButtons = useMemo(() => {
    const buttons = [];
    const maxPageButtons = pageButtonsPerSide * 2 + 1;
    let startPage = Math.max(currentPage - Math.floor(maxPageButtons / 2), 0);
    const endPage = Math.min(startPage + maxPageButtons - 1, pages - 1);

    if (endPage - startPage < maxPageButtons - 1) {
      startPage = Math.max(endPage - maxPageButtons + 1, 0);
    }

    for (let i = startPage; i <= endPage; i++) {
      buttons.push(i);
    }

    return buttons;
  }, [currentPage, pageButtonsPerSide, pages]);

  const isFirstEnabled = useMemo(() => currentPage > 0, [currentPage]);
  const isLastEnabled = useMemo(() => currentPage < pages - 1, [currentPage, pages]);

  const rangeDisplay = useMemo(() => {
    const startItem = currentPage * (limit || 0) + 1;
    const endItem = Math.min((currentPage + 1) * (limit || 0), total);
    return `${startItem}-${endItem} of ${total}`;
  }, [total, currentPage, limit]);

  const api = useMemo(
    () => ({
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
    }),
    [
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
    ]
  );

  return api;
};
