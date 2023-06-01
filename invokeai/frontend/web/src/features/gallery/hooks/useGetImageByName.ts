import { useAppSelector } from 'app/store/storeHooks';
import { selectImagesEntities } from '../store/imagesSlice';
import { useCallback } from 'react';

const useGetImageByName = () => {
  const images = useAppSelector(selectImagesEntities);
  return useCallback(
    (name: string | undefined) => {
      if (!name) {
        return;
      }
      return images[name];
    },
    [images]
  );
};

export default useGetImageByName;
