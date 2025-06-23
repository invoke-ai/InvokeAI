import { Flex, Spacer } from '@invoke-ai/ui-library';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { memo } from 'react';

import { CurrentImageButtons } from './CurrentImageButtons';
import { ToggleProgressButton } from './ToggleProgressButton';

export const ViewerToolbar = memo(() => {
  return (
    <Flex w="full" justifyContent="center" h={8}>
      <ToggleProgressButton />
      <Spacer />
      <CurrentImageButtons />
      <Spacer />
      <ToggleMetadataViewerButton />
    </Flex>
  );
});

ViewerToolbar.displayName = 'ViewerToolbar';
