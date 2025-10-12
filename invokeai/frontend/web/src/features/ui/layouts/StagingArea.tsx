import { Flex } from '@invoke-ai/ui-library';
import { StagingAreaItemsList } from 'features/controlLayers/components/StagingArea/StagingAreaItemsList';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { useActiveCanvasIsStaging } from 'features/controlLayers/hooks/useCanvasIsStaging';
import { memo } from 'react';

export const StagingArea = memo(() => {
  const isStaging = useActiveCanvasIsStaging();

  if (!isStaging) {
    return null;
  }

  return (
    <Flex position="absolute" flexDir="column" bottom={2} gap={2} align="center" justify="center" left={2} right={2}>
      <StagingAreaItemsList />
      <StagingAreaToolbar />
    </Flex>
  );
});
StagingArea.displayName = 'StagingArea';
