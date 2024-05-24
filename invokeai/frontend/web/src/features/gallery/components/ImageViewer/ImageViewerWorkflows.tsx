import { Flex } from '@invoke-ai/ui-library';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { memo } from 'react';

import CurrentImageButtons from './CurrentImageButtons';
import CurrentImagePreview from './CurrentImagePreview';

export const ImageViewerWorkflows = memo(() => {
  return (
    <Flex
      layerStyle="first"
      borderRadius="base"
      position="absolute"
      flexDirection="column"
      top={0}
      right={0}
      bottom={0}
      left={0}
      p={2}
      rowGap={4}
      alignItems="center"
      justifyContent="center"
      zIndex={10} // reactflow puts its minimap at 5, so we need to be above that
    >
      <Flex w="full" gap={2}>
        <Flex flex={1} justifyContent="center">
          <Flex gap={2} marginInlineEnd="auto">
            <ToggleProgressButton />
            <ToggleMetadataViewerButton />
          </Flex>
        </Flex>
        <Flex flex={1} gap={2} justifyContent="center">
          <CurrentImageButtons />
        </Flex>
        <Flex flex={1} justifyContent="center">
          <Flex gap={2} marginInlineStart="auto" />
        </Flex>
      </Flex>
      <CurrentImagePreview />
    </Flex>
  );
});

ImageViewerWorkflows.displayName = 'ImageViewerWorkflows';
