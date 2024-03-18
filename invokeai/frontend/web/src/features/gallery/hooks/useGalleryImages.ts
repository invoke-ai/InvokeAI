import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { moreImagesLoaded } from 'features/gallery/store/gallerySlice';
import { useCallback, useMemo } from 'react';
import { useGetBoardAssetsTotalQuery, useGetBoardImagesTotalQuery } from 'services/api/endpoints/boards';
import { useListImagesQuery } from 'services/api/endpoints/images';

/**
 * Provides access to the gallery images and a way to imperatively fetch more.
 */
export const useGalleryImages = () => {
  const dispatch = useAppDispatch();
  const galleryView = useAppSelector((s) => s.gallery.galleryView);
  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const queryResult = useListImagesQuery(queryArgs);
  const selectedBoardId = useAppSelector((s) => s.gallery.selectedBoardId);
  const { data: assetsTotal } = useGetBoardAssetsTotalQuery(selectedBoardId);
  const { data: imagesTotal } = useGetBoardImagesTotalQuery(selectedBoardId);
  const currentViewTotal = useMemo(
    () => (galleryView === 'images' ? imagesTotal?.total : assetsTotal?.total),
    [assetsTotal?.total, galleryView, imagesTotal?.total]
  );
  const areMoreImagesAvailable = useMemo(() => {
    if (!currentViewTotal || !queryResult.data) {
      return false;
    }
    return queryResult.data.ids.length < currentViewTotal;
  }, [queryResult.data, currentViewTotal]);
  const handleLoadMoreImages = useCallback(() => {
    dispatch(moreImagesLoaded());
  }, [dispatch]);

  return {
    areMoreImagesAvailable,
    handleLoadMoreImages,
    queryResult,
  };
};
