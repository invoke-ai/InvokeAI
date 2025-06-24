import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectSearchTerm } from 'features/gallery/store/gallerySelectors';
import { searchTermChanged } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';

export const useGallerySearchTerm = () => {
  // Highlander!
  // useAssertSingleton('gallery-search-state');

  const dispatch = useAppDispatch();
  const searchTerm = useAppSelector(selectSearchTerm);

  const onChange = useCallback(
    (val: string) => {
      dispatch(searchTermChanged(val));
    },
    [dispatch]
  );

  const onReset = useCallback(() => {
    dispatch(searchTermChanged(''));
  }, [dispatch]);

  return [searchTerm, onChange, onReset] as const;
};
