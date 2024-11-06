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
    <Flex w="full" px={2} gap={2} bg="base.750" borderTopRadius="base" h={12}>
      <Flex flex={1} justifyContent="center">
        <Flex marginInlineEnd="auto" alignItems="center">
          <ToggleProgressButton />
          <ToggleMetadataViewerButton />
        </Flex>
      </Flex>
      <Flex flex={1} justifyContent="center" alignItems="center">
        <CurrentImageButtons />
      </Flex>
      <Flex flex={1} justifyContent="center">
        <Flex marginInlineStart="auto" alignItems="center">
          {closeButton}
        </Flex>
      </Flex>
    </Flex>
  );
});

ViewerToolbar.displayName = 'ViewerToolbar';
