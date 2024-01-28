import { Flex } from '@invoke-ai/ui-library';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';
import { memo } from 'react';

export const Prompts = memo(() => {
  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      <ParamNegativePrompt />
    </Flex>
  );
});

Prompts.displayName = 'Prompts';
