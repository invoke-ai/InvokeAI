import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGalleryItemDTO } from 'common/hooks/useGalleryItemDTO';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';

import { CurrentImageButtons } from './CurrentImageButtons';
import { ToggleProgressButton } from './ToggleProgressButton';

export const ImageViewerToolbar = memo(() => {
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const galleryItem = useGalleryItemDTO(lastSelectedItem);

  // Videos don't carry workflows or recallable metadata yet — the action row + metadata viewer
  // toggle are image-specific. We still show the progress button (it's media-agnostic).
  const showImageActions = galleryItem?.kind === 'image';

  return (
    <Flex w="full" justifyContent="center" h={8}>
      <ToggleProgressButton />
      <Spacer />
      {showImageActions && <CurrentImageButtons imageDTO={galleryItem.dto} />}
      <Spacer />
      {showImageActions && <ToggleMetadataViewerButton />}
    </Flex>
  );
});

ImageViewerToolbar.displayName = 'ImageViewerToolbar';
