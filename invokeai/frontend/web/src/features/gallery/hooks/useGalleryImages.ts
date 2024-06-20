import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { offsetChanged } from 'features/gallery/store/gallerySlice';
import { useCallback, useMemo } from 'react';
import { useGetBoardAssetsTotalQuery, useGetBoardImagesTotalQuery } from 'services/api/endpoints/boards';
import { useListImagesQuery } from 'services/api/endpoints/images';

const LIMIT = 20;

export const useGalleryImages = () => {
  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const queryResult = useListImagesQuery(queryArgs);
  const imageDTOs = useMemo(() => queryResult.data?.items ?? EMPTY_ARRAY, [queryResult.data]);
  return {
    imageDTOs,
    queryResult,
  };
};

export const useGalleryPagination = () => {
  const dispatch = useAppDispatch();
  const offset = useAppSelector((s) => s.gallery.offset);
  const galleryView = useAppSelector((s) => s.gallery.galleryView);
  const selectedBoardId = useAppSelector((s) => s.gallery.selectedBoardId);
  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const { count } = useListImagesQuery(queryArgs, {
    selectFromResult: ({ data }) => ({ count: data?.items.length ?? 0 }),
  });
  const { data: assetsTotal } = useGetBoardAssetsTotalQuery(selectedBoardId);
  const { data: imagesTotal } = useGetBoardImagesTotalQuery(selectedBoardId);
  const total = useMemo(() => {
    if (galleryView === 'images') {
      return imagesTotal?.total ?? 0;
    } else {
      return assetsTotal?.total ?? 0;
    }
  }, [assetsTotal?.total, galleryView, imagesTotal?.total]);
  const page = useMemo(() => Math.floor(offset / LIMIT), [offset]);
  const pages = useMemo(() => Math.floor(total / LIMIT), [total]);
  const isNextEnabled = useMemo(() => {
    if (!count) {
      return false;
    }
    return page < pages;
  }, [count, page, pages]);
  const isPrevEnabled = useMemo(() => {
    if (!count) {
      return false;
    }
    return offset > 0;
  }, [count, offset]);
  const next = useCallback(() => {
    dispatch(offsetChanged(offset + LIMIT));
  }, [dispatch, offset]);
  const prev = useCallback(() => {
    dispatch(offsetChanged(Math.max(offset - LIMIT, 0)));
  }, [dispatch, offset]);
  const goToPage = useCallback(
    (page: number) => {
      const p = Math.max(0, Math.min(page, pages - 1));
      dispatch(offsetChanged(p));
    },
    [dispatch, pages]
  );
  const first = useCallback(() => {
    dispatch(offsetChanged(0));
  }, [dispatch]);
  const last = useCallback(() => {
    dispatch(offsetChanged(pages * LIMIT));
  }, [dispatch, pages]);
  // calculate the page buttons to display - current page with 3 around it
  const pageButtons = useMemo(() => {
    const buttons = [];
    const start = Math.max(0, page - 3);
    const end = Math.min(pages, start + 6);
    for (let i = start; i < end; i++) {
      buttons.push(i);
    }
    return buttons;
  }, [page, pages]);
  const isFirstEnabled = useMemo(() => page > 0, [page]);
  const isLastEnabled = useMemo(() => page < pages - 1, [page, pages]);

  const api = useMemo(
    () => ({
      count,
      total,
      page,
      pages,
      isNextEnabled,
      isPrevEnabled,
      next,
      prev,
      goToPage,
      first,
      last,
      pageButtons,
      isFirstEnabled,
      isLastEnabled,
    }),
    [
      count,
      total,
      page,
      pages,
      isNextEnabled,
      isPrevEnabled,
      next,
      prev,
      goToPage,
      first,
      last,
      pageButtons,
      isFirstEnabled,
      isLastEnabled,
    ]
  );
  return api;
};