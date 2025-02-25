import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectListAllImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';
import { useListImagesQuery } from 'services/api/endpoints/images';

export const useSelectAll = () => {
  const dispatch = useAppDispatch();
  const queryArgs = useAppSelector(selectListAllImagesQueryArgs);
  const { data } = useListImagesQuery(queryArgs);

  return useCallback(() => {
    if (data) {
      dispatch(selectionChanged(data.items));
    }
  }, [dispatch, data]);
};
