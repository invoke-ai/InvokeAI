import { skipToken } from '@reduxjs/toolkit/query';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { selectGetImageNamesQueryArgs, selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { getDateFromVirtualBoardId, isVirtualBoardId } from 'features/gallery/store/types';
import { useMemo } from 'react';
import { useGetGalleryItemNamesQuery } from 'services/api/endpoints/gallery';
import { useGetVirtualBoardImageNamesByDateQuery } from 'services/api/endpoints/virtual_boards';
import { useDebounce } from 'use-debounce';

const selectFromGalleryItemNamesResult = ({
  currentData,
  isLoading,
  isFetching,
}: {
  currentData?: { items: { kind: 'image' | 'video'; name: string }[] };
  isLoading: boolean;
  isFetching: boolean;
}) => ({
  items: currentData?.items ?? (EMPTY_ARRAY as { kind: 'image' | 'video'; name: string }[]),
  isLoading,
  isFetching,
});

const selectFromVirtualBoardResult = ({
  currentData,
  isLoading,
  isFetching,
}: {
  currentData?: { image_names: string[] };
  isLoading: boolean;
  isFetching: boolean;
}) => ({
  imageNames: currentData?.image_names ?? EMPTY_ARRAY,
  isLoading,
  isFetching,
});

const galleryQueryOptions = {
  refetchOnReconnect: true,
  selectFromResult: selectFromGalleryItemNamesResult,
};

const virtualBoardQueryOptions = {
  refetchOnReconnect: true,
  selectFromResult: selectFromVirtualBoardResult,
};

/**
 * Returns the ordered flat list of gallery item names. Names are polymorphic — both image and
 * video names appear in the same list, interleaved by created_at. Callers that need to know the
 * kind of a particular name use `isVideoName` from `features/gallery/store/types`.
 *
 * Virtual boards (date-based) are image-only for now and call the legacy by-date endpoint.
 */
export const useGalleryImageNames = () => {
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const _imageQueryArgs = useAppSelector(selectGetImageNamesQueryArgs);
  const [imageQueryArgs] = useDebounce(_imageQueryArgs, 300);
  const isVirtual = isVirtualBoardId(selectedBoardId);

  // The polymorphic gallery names endpoint shares the same filter args as the image names
  // endpoint (board_id, categories, search_term, order_dir, starred_first, is_intermediate).
  const galleryResult = useGetGalleryItemNamesQuery(isVirtual ? skipToken : imageQueryArgs, galleryQueryOptions);

  const date = isVirtual ? getDateFromVirtualBoardId(selectedBoardId) : '';
  const virtualResult = useGetVirtualBoardImageNamesByDateQuery(
    isVirtual
      ? {
          date,
          categories: imageQueryArgs.categories ?? undefined,
          search_term: imageQueryArgs.search_term || undefined,
          order_dir: imageQueryArgs.order_dir,
          starred_first: imageQueryArgs.starred_first,
        }
      : skipToken,
    virtualBoardQueryOptions
  );

  // Flat names + isLoading exposed for backward compatibility with the existing callers (paged
  // grid, search, navigation hotkeys). The kind is recoverable from the filename extension.
  const imageNames = useMemo(() => {
    if (isVirtual) {
      return virtualResult.imageNames;
    }
    return galleryResult.items.map((ref) => ref.name);
  }, [isVirtual, virtualResult.imageNames, galleryResult.items]);

  return {
    imageNames,
    isLoading: isVirtual ? virtualResult.isLoading : galleryResult.isLoading,
    isFetching: isVirtual ? virtualResult.isFetching : galleryResult.isFetching,
    queryArgs: imageQueryArgs,
  };
};
