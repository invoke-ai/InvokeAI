import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useVideoDTO } from 'services/api/endpoints/videos';

import { CurrentVideoButtons } from './CurrentVideoButtons';

export const VideoViewerToolbar = memo(() => {
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const videoDTO = useVideoDTO(lastSelectedItem?.type === 'video' ? lastSelectedItem.id : null);

  return (
    <Flex w="full" justifyContent="center" h={8}>
      {videoDTO && <CurrentVideoButtons videoDTO={videoDTO} />}
      {videoDTO && <ToggleMetadataViewerButton />}
    </Flex>
  );
});

VideoViewerToolbar.displayName = 'VideoViewerToolbar';
