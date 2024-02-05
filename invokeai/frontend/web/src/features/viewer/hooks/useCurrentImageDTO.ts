import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { selectLatestImageName } from 'features/viewer/hooks/useLatestImageDTO';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

/**
 *  The currently displayed image's DTO. If the viewer mode is 'progress', returns the latest image's DTO.
 */
export const useCurrentImageDTO = () => {
  const latestImageName = useAppSelector(selectLatestImageName);
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const viewerMode = useAppSelector((s) => s.viewer.viewerMode);
  const { currentData: imageDTO } = useGetImageDTOQuery(
    (viewerMode === 'progress' ? latestImageName : lastSelectedImage?.image_name) ?? skipToken
  );

  return imageDTO;
};
