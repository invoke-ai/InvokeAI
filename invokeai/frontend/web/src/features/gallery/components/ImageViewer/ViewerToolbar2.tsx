import { ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { memo } from 'react';

import CurrentImageButtons from './CurrentImageButtons';

export const ViewerToolbar = memo(() => {
  return (
    <Flex w="full" justifyContent="center" h="24px">
      <ButtonGroup>
        <ToggleMetadataViewerButton />
        <CurrentImageButtons />
      </ButtonGroup>
    </Flex>
  );
});

ViewerToolbar.displayName = 'ViewerToolbar';
