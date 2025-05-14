import { Flex } from '@invoke-ai/ui-library';
import { SimpleTabAspectRatio } from 'features/simpleGeneration/components/SimpleTabAspectRatio';
import { SimpleTabModel } from 'features/simpleGeneration/components/SImpleTabModel';
import { SimpleTabPositivePrompt } from 'features/simpleGeneration/components/SimpleTabPositivePrompt';
import { memo } from 'react';

export const SimpleTabLeftPanel = memo(() => {
  return (
    <Flex w="full" h="full" flexDir="column" gap={2}>
      <SimpleTabPositivePrompt />
      <SimpleTabModel />
      <SimpleTabAspectRatio />
    </Flex>
  );
});

SimpleTabLeftPanel.displayName = 'SimpleTabLeftPanel';
