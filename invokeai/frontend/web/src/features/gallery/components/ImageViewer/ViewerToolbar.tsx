import { ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import type { ReactNode } from 'react';
import { memo } from 'react';

import CurrentImageButtons from './CurrentImageButtons';

type Props = {
  closeButton?: ReactNode;
};

export const ViewerToolbar = memo(({ closeButton }: Props) => {
  return (
    <Flex w='full' justifyContent="center" h="24px">
      <ButtonGroup>
        <ToggleMetadataViewerButton />
        <CurrentImageButtons />
      </ButtonGroup>
    </Flex>
  );
});

ViewerToolbar.displayName = 'ViewerToolbar';
