import { Divider, Flex } from '@invoke-ai/ui-library';
import { StagingAreaHeader } from 'features/controlLayers/components/SimpleSession/StagingAreaHeader';
import { StagingAreaNoItems } from 'features/controlLayers/components/SimpleSession/StagingAreaNoItems';
import { memo } from 'react';

export const SimpleSessionNoId = memo(() => {
  return (
    <Flex flexDir="column" gap={2} w="full" h="full" minW={0} minH={0}>
      <StagingAreaHeader />
      <Divider />
      <StagingAreaNoItems />
    </Flex>
  );
});
SimpleSessionNoId.displayName = 'StSimpleSessionNoIdagingArea';
