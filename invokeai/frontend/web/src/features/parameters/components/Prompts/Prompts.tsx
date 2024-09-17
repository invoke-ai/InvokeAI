import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { createParamsSelector, selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';
import { ParamSDXLNegativeStylePrompt } from 'features/sdxl/components/SDXLPrompts/ParamSDXLNegativeStylePrompt';
import { ParamSDXLPositiveStylePrompt } from 'features/sdxl/components/SDXLPrompts/ParamSDXLPositiveStylePrompt';
import { memo } from 'react';

const selectWithStylePrompts = createParamsSelector((params) => {
  const isSDXL = params.model?.base === 'sdxl';
  const shouldConcatPrompts = params.shouldConcatPrompts;
  return isSDXL && !shouldConcatPrompts;
});

export const Prompts = memo(() => {
  const withStylePrompts = useAppSelector(selectWithStylePrompts);
  const isFLUX = useAppSelector(selectIsFLUX);
  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      {withStylePrompts && <ParamSDXLPositiveStylePrompt />}
      {!isFLUX && <ParamNegativePrompt />}
      {withStylePrompts && <ParamSDXLNegativeStylePrompt />}
    </Flex>
  );
});

Prompts.displayName = 'Prompts';
