import { Flex } from '@invoke-ai/ui-library';
import { ViewerToolbarImageButtons } from 'features/viewer/components/ViewerToolbarImageButtons';
import { ViewerToolbarImageMenu } from 'features/viewer/components/ViewerToolbarImageMenu';
import { ViewerToolbarModeButtons } from 'features/viewer/components/ViewerToolbarModeButtons';
import { memo } from 'react';

export const ViewerToolbar = memo(() => {
  return (
    <Flex flexWrap="wrap" justifyContent="center" alignItems="center" gap={2} w="full">
      <Flex flexGrow={1} flexWrap="wrap" justifyContent="flex-start" alignItems="center" gap={2}>
        <ViewerToolbarImageMenu />
      </Flex>
      <Flex flexGrow={1} flexWrap="wrap" justifyContent="center" alignItems="center" gap={2}>
        <ViewerToolbarImageButtons />
      </Flex>
      <Flex flexGrow={1} flexWrap="wrap" justifyContent="flex-end" alignItems="center" gap={2}>
        <ViewerToolbarModeButtons />
      </Flex>
    </Flex>
  );
});

ViewerToolbar.displayName = 'ViewerToolbar';
