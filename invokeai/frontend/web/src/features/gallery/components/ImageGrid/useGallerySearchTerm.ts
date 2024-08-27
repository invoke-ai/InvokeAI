import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectSearchTerm } from 'features/gallery/store/gallerySelectors';
import { searchTermChanged } from 'features/gallery/store/gallerySlice';
import { debounce } from 'lodash-es';
import { useCallback, useMemo, useState } from 'react';

export const useGallerySearchTerm = () => {
  // Highlander!
  useAssertSingleton('gallery-search-state');

  const dispatch = useAppDispatch();
  const searchTerm = useAppSelector(selectSearchTerm);

  const [localSearchTerm, setLocalSearchTerm] = useState(searchTerm);

  const debouncedSetSearchTerm = useMemo(() => {
    return debounce((val: string) => {
      dispatch(searchTermChanged(val));
    }, 1000);
  }, [dispatch]);

  const onChange = useCallback(
    (val: string) => {
      setLocalSearchTerm(val);
      debouncedSetSearchTerm(val);
    },
    [debouncedSetSearchTerm]
  );

  const onReset = useCallback(() => {
    debouncedSetSearchTerm.cancel();
    setLocalSearchTerm('');
    dispatch(searchTermChanged(''));
  }, [debouncedSetSearchTerm, dispatch]);

  return [localSearchTerm, onChange, onReset] as const;
};
