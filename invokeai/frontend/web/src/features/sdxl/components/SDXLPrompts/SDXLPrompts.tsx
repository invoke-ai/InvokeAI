import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';
import { memo } from 'react';

import { ParamSDXLNegativeStylePrompt } from './ParamSDXLNegativeStylePrompt';
import { ParamSDXLPositiveStylePrompt } from './ParamSDXLPositiveStylePrompt';

export const SDXLPrompts = memo(() => {
  const shouldConcatPrompts = useAppSelector((s) => s.controlLayers.present.shouldConcatPrompts);
  return (
    <Flex flexDir="column" gap={2} pos="relative">
      <ParamPositivePrompt />
      {!shouldConcatPrompts && <ParamSDXLPositiveStylePrompt />}
      <ParamNegativePrompt />
      {!shouldConcatPrompts && <ParamSDXLNegativeStylePrompt />}
    </Flex>
  );
});

SDXLPrompts.displayName = 'SDXLPrompts';
