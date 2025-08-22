import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useImageDTO } from 'services/api/endpoints/images';

import { CurrentImageButtons } from './CurrentImageButtons';
import { ToggleProgressButton } from './ToggleProgressButton';

export const ImageViewerToolbar = memo(() => {
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const imageDTO = useImageDTO(lastSelectedItem?.id);

  return (
    <Flex w="full" justifyContent="center" h={8}>
      <ToggleProgressButton />
      <Spacer />
      {imageDTO && <CurrentImageButtons imageDTO={imageDTO} />}
      <Spacer />
      {imageDTO && <ToggleMetadataViewerButton />}
    </Flex>
  );
});

ImageViewerToolbar.displayName = 'ImageViewerToolbar';
