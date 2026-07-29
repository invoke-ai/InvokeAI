import { useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { selectLastSelectedItem, selectSelectionCount } from 'features/gallery/store/gallerySelectors';
import { isVideoName } from 'features/gallery/store/types';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback } from 'react';
import { useImageDTO, useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';
import { useStarVideosMutation, useUnstarVideosMutation, useVideoDTO } from 'services/api/endpoints/videos';

export const useGalleryStarImageHotkey = () => {
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const selectionCount = useAppSelector(selectSelectionCount);
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isVideo = lastSelectedItem ? isVideoName(lastSelectedItem) : false;
  const imageDTO = useImageDTO(isVideo ? null : lastSelectedItem);
  const videoDTO = useVideoDTO(isVideo ? lastSelectedItem : null);
  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();
  const [starVideos] = useStarVideosMutation();
  const [unstarVideos] = useUnstarVideosMutation();

  const dto = isVideo ? videoDTO : imageDTO;

  const handleStarHotkey = useCallback(() => {
    if (!isGalleryFocused) {
      return;
    }
    if (isVideo) {
      if (!videoDTO) {
        return;
      }
      if (videoDTO.starred) {
        unstarVideos({ video_names: [videoDTO.video_name] });
      } else {
        starVideos({ video_names: [videoDTO.video_name] });
      }
    } else {
      if (!imageDTO) {
        return;
      }
      if (imageDTO.starred) {
        unstarImages({ image_names: [imageDTO.image_name] });
      } else {
        starImages({ image_names: [imageDTO.image_name] });
      }
    }
  }, [imageDTO, isGalleryFocused, isVideo, starImages, starVideos, unstarImages, unstarVideos, videoDTO]);

  useRegisteredHotkeys({
    id: 'starImage',
    category: 'gallery',
    callback: handleStarHotkey,
    options: { enabled: !!dto && selectionCount === 1 && isGalleryFocused },
    dependencies: [dto, selectionCount, isGalleryFocused, handleStarHotkey],
  });
};
