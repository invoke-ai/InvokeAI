import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { StagingAreaItemsList } from 'features/controlLayers/components/SimpleSession/StagingAreaItemsList';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { memo } from 'react';

export const StagingArea = memo(() => {
  const ctx = useCanvasSessionContext();
  const hasItems = useStore(ctx.$hasItems);

  if (!hasItems) {
    return null;
  }

  return (
    <Flex position="absolute" flexDir="column" bottom={4} gap={2} align="center" justify="center" left={4} right={4}>
      <StagingAreaItemsList />
      <StagingAreaToolbar />
    </Flex>
  );
});
StagingArea.displayName = 'StagingArea';
