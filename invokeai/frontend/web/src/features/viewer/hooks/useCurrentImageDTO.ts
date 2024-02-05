import { useAppSelector } from 'app/store/storeHooks';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { useLatestImageDTO } from 'features/viewer/hooks/useLatestImageDTO';
import { useMemo } from 'react';

/**
 *  The currently displayed image's DTO. If the viewer mode is 'progress', returns the latest image's DTO.
 */
export const useCurrentImageDTO = () => {
  const latestImageDTO = useLatestImageDTO();
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const viewerMode = useAppSelector((s) => s.viewer.viewerMode);
  const imageDTO = useMemo(() => {
    if (viewerMode === 'progress') {
      return latestImageDTO;
    }
    return lastSelectedImage;
  }, [latestImageDTO, lastSelectedImage, viewerMode]);

  return imageDTO;
};
