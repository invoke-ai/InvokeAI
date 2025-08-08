import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { RefImageList } from 'features/controlLayers/components/RefImage/RefImageList';
import { selectHasNegativePrompt, selectModelSupportsNegativePrompt } from 'features/controlLayers/store/paramsSlice';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';
import { memo } from 'react';

export const Prompts = memo(() => {
  const modelSupportsNegativePrompt = useAppSelector(selectModelSupportsNegativePrompt);
  const hasNegativePrompt = useAppSelector(selectHasNegativePrompt);

  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      {modelSupportsNegativePrompt && hasNegativePrompt && <ParamNegativePrompt />}
      <RefImageList />
    </Flex>
  );
});

Prompts.displayName = 'Prompts';
