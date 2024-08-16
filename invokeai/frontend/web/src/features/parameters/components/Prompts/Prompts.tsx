import { Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';
import { ParamSDXLNegativeStylePrompt } from 'features/sdxl/components/SDXLPrompts/ParamSDXLNegativeStylePrompt';
import { ParamSDXLPositiveStylePrompt } from 'features/sdxl/components/SDXLPrompts/ParamSDXLPositiveStylePrompt';
import { memo } from 'react';

const concatPromptsSelector = createSelector(
  [selectCanvasV2Slice],
  (canvasV2) => {
    return canvasV2.params.model?.base !== 'sdxl' || canvasV2.params.shouldConcatPrompts;
  }
);

export const Prompts = memo(() => {
  const shouldConcatPrompts = useAppSelector(concatPromptsSelector);
  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      {!shouldConcatPrompts && <ParamSDXLPositiveStylePrompt />}
      <ParamNegativePrompt />
      {!shouldConcatPrompts && <ParamSDXLNegativeStylePrompt />}
    </Flex>
  );
});

Prompts.displayName = 'Prompts';
