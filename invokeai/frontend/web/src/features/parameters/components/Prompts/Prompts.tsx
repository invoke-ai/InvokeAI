import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';
import { ParamSDXLNegativeStylePrompt } from 'features/sdxl/components/SDXLPrompts/ParamSDXLNegativeStylePrompt';
import { ParamSDXLPositiveStylePrompt } from 'features/sdxl/components/SDXLPrompts/ParamSDXLPositiveStylePrompt';
import { memo } from 'react';

export const Prompts = memo(() => {
  const withStylePrompts = useAppSelector((s) => {
    const isSDXL = s.params.model?.base === 'sdxl';
    const shouldConcatPrompts = s.params.shouldConcatPrompts;
    return isSDXL && !shouldConcatPrompts;
  });
  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      {withStylePrompts && <ParamSDXLPositiveStylePrompt />}
      <ParamNegativePrompt />
      {withStylePrompts && <ParamSDXLNegativeStylePrompt />}
    </Flex>
  );
});

Prompts.displayName = 'Prompts';
