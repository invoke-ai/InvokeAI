import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectHasNegativePrompt, selectModelSupportsNegativePrompt } from 'features/controlLayers/store/paramsSlice';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';
import { memo } from 'react';

export const UpscalePrompts = memo(() => {
  const modelSupportsNegativePrompt = useAppSelector(selectModelSupportsNegativePrompt);
  const hasNegativePrompt = useAppSelector(selectHasNegativePrompt);
  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      {modelSupportsNegativePrompt && hasNegativePrompt && <ParamNegativePrompt />}
    </Flex>
  );
});

UpscalePrompts.displayName = 'UpscalePrompts';
