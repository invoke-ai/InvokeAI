import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { RefImageList } from 'features/controlLayers/components/RefImage/RefImageList';
import {
  selectHasNegativePrompt,
  selectModelSupportsNegativePrompt,
  selectModelSupportsRefImages,
} from 'features/controlLayers/store/paramsSlice';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';
import { memo } from 'react';

export const Prompts = memo(() => {
  const modelSupportsNegativePrompt = useAppSelector(selectModelSupportsNegativePrompt);
  const modelSupportsRefImages = useAppSelector(selectModelSupportsRefImages);
  const hasNegativePrompt = useAppSelector(selectHasNegativePrompt);

  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      {modelSupportsNegativePrompt && hasNegativePrompt && <ParamNegativePrompt />}
      {modelSupportsRefImages && <RefImageList />}
    </Flex>
  );
});

Prompts.displayName = 'Prompts';
