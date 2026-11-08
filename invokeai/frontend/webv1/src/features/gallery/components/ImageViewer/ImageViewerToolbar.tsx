import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGalleryItemDTO } from 'common/hooks/useGalleryItemDTO';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';

import { CurrentImageButtons } from './CurrentImageButtons';
import { CurrentVideoButtons } from './CurrentVideoButtons';
import { ToggleProgressButton } from './ToggleProgressButton';

export const ImageViewerToolbar = memo(() => {
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const galleryItem = useGalleryItemDTO(lastSelectedItem);

  // Images get the full action row; videos get their own trimmed row (load workflow). The
  // metadata viewer toggle works for both kinds, and the progress button is media-agnostic.
  const showImageActions = galleryItem?.kind === 'image';
  const showVideoActions = galleryItem?.kind === 'video';

  return (
    <Flex w="full" justifyContent="center" h={8}>
      <ToggleProgressButton />
      <Spacer />
      {showImageActions && <CurrentImageButtons imageDTO={galleryItem.dto} />}
      {showVideoActions && <CurrentVideoButtons videoDTO={galleryItem.dto} />}
      <Spacer />
      {(showImageActions || showVideoActions) && <ToggleMetadataViewerButton />}
    </Flex>
  );
});

ImageViewerToolbar.displayName = 'ImageViewerToolbar';
