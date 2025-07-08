import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  createParamsSelector,
  selectHasNegativePrompt,
  selectModelSupportsNegativePrompt,
} from 'features/controlLayers/store/paramsSlice';
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

export const UpscalePrompts = memo(() => {
  const withStylePrompts = useAppSelector(selectWithStylePrompts);
  const modelSupportsNegativePrompt = useAppSelector(selectModelSupportsNegativePrompt);
  const hasNegativePrompt = useAppSelector(selectHasNegativePrompt);
  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      {withStylePrompts && <ParamSDXLPositiveStylePrompt />}
      {modelSupportsNegativePrompt && hasNegativePrompt && <ParamNegativePrompt />}
      {withStylePrompts && <ParamSDXLNegativeStylePrompt />}
    </Flex>
  );
});

UpscalePrompts.displayName = 'UpscalePrompts'; 