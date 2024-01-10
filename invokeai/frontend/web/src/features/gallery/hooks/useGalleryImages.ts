import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { selectListImagesBaseQueryArgs } from 'features/gallery/store/gallerySelectors';
import { IMAGE_LIMIT } from 'features/gallery/store/types';
import { atom } from 'nanostores';
import { useCallback, useMemo } from 'react';
import {
  useGetBoardAssetsTotalQuery,
  useGetBoardImagesTotalQuery,
} from 'services/api/endpoints/boards';
import { useListImagesQuery } from 'services/api/endpoints/images';
import type { ListImagesArgs } from 'services/api/types';

// The gallery is a singleton but multiple components need access to its query data.
// If we don't define the query args outside of the hook, then each component will
// have its own query args and trigger multiple requests. We use an atom to store
// the query args outside of the hook so that all consumers use the same query args.
const $queryArgs = atom<ListImagesArgs | null>(null);

/**
 * Provides access to the gallery images and a way to imperatively fetch more.
 *
 * This hook is a singleton.
 */
export const useGalleryImages = () => {
  const galleryView = useAppSelector((s) => s.gallery.galleryView);
  const baseQueryArgs = useAppSelector(selectListImagesBaseQueryArgs);
  const queryArgs = useStore($queryArgs);
  const queryResult = useListImagesQuery(queryArgs ?? baseQueryArgs);
  const boardId = useMemo(
    () => baseQueryArgs.board_id ?? 'none',
    [baseQueryArgs.board_id]
  );
  const { data: assetsTotal } = useGetBoardAssetsTotalQuery(boardId);
  const { data: imagesTotal } = useGetBoardImagesTotalQuery(boardId);
  const currentViewTotal = useMemo(
    () => (galleryView === 'images' ? imagesTotal?.total : assetsTotal?.total),
    [assetsTotal?.total, galleryView, imagesTotal?.total]
  );
  const loadedImagesCount = useMemo(
    () => queryResult.data?.ids.length ?? 0,
    [queryResult.data?.ids.length]
  );
  const areMoreImagesAvailable = useMemo(() => {
    if (!currentViewTotal || !queryResult.data) {
      return false;
    }
    return queryResult.data.ids.length < currentViewTotal;
  }, [queryResult.data, currentViewTotal]);
  const handleLoadMoreImages = useCallback(() => {
    // To load more images, we update the query args with an offset and limit.
    const _queryArgs: ListImagesArgs = loadedImagesCount
      ? {
          ...baseQueryArgs,
          offset: loadedImagesCount,
          limit: IMAGE_LIMIT,
        }
      : baseQueryArgs;
    $queryArgs.set(_queryArgs);
  }, [baseQueryArgs, loadedImagesCount]);

  return {
    areMoreImagesAvailable,
    handleLoadMoreImages,
    queryResult,
  };
};
