import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useImageDTO } from 'services/api/endpoints/images';

import { CurrentImageButtons } from './CurrentImageButtons';
import { ToggleProgressButton } from './ToggleProgressButton';

export const ViewerToolbar = memo(() => {
  const imageName = useAppSelector(selectLastSelectedImage);
  const imageDTO = useImageDTO(imageName);

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

ViewerToolbar.displayName = 'ViewerToolbar';
