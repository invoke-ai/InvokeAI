import { Flex } from '@invoke-ai/ui-library';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import type { ReactNode } from 'react';
import { memo } from 'react';

import CurrentImageButtons from './CurrentImageButtons';

type Props = {
  closeButton?: ReactNode;
};

export const ViewerToolbar = memo(({ closeButton }: Props) => {
  return (
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
        <Flex gap={2} marginInlineStart="auto">
          {closeButton}
        </Flex>
      </Flex>
    </Flex>
  );
});

ViewerToolbar.displayName = 'ViewerToolbar';
