import { useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { selectLastSelectedItem, selectSelectionCount } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback } from 'react';
import { useImageDTO, useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';

export const useGalleryStarImageHotkey = () => {
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const selectionCount = useAppSelector(selectSelectionCount);
  const isGalleryFocused = useIsRegionFocused('gallery');
  const imageDTO = useImageDTO(lastSelectedItem);
  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const handleStarHotkey = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!isGalleryFocused) {
      return;
    }
    if (imageDTO.starred) {
      unstarImages({ image_names: [imageDTO.image_name] });
    } else {
      starImages({ image_names: [imageDTO.image_name] });
    }
  }, [imageDTO, isGalleryFocused, starImages, unstarImages]);

  useRegisteredHotkeys({
    id: 'starImage',
    category: 'gallery',
    callback: handleStarHotkey,
    options: { enabled: !!imageDTO && selectionCount === 1 && isGalleryFocused },
    dependencies: [imageDTO, selectionCount, isGalleryFocused, handleStarHotkey],
  });
};
