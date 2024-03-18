import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';
import { memo } from 'react';

import { ParamSDXLNegativeStylePrompt } from './ParamSDXLNegativeStylePrompt';
import { ParamSDXLPositiveStylePrompt } from './ParamSDXLPositiveStylePrompt';

export const SDXLPrompts = memo(() => {
  const shouldConcatSDXLStylePrompt = useAppSelector((s) => s.sdxl.shouldConcatSDXLStylePrompt);
  return (
    <Flex flexDir="column" gap={2} pos="relative">
      <ParamPositivePrompt />
      {!shouldConcatSDXLStylePrompt && <ParamSDXLPositiveStylePrompt />}
      <ParamNegativePrompt />
      {!shouldConcatSDXLStylePrompt && <ParamSDXLNegativeStylePrompt />}
    </Flex>
  );
});

SDXLPrompts.displayName = 'SDXLPrompts';
