import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { useMemo } from 'react';
import { useListImagesQuery } from 'services/api/endpoints/images';

export const useGalleryImages = () => {
  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const queryResult = useListImagesQuery(queryArgs);
  const imageDTOs = useMemo(() => queryResult.data?.items ?? EMPTY_ARRAY, [queryResult.data]);
  return {
    imageDTOs,
    queryResult,
  };
};
