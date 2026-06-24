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
import { memo, useMemo } from 'react';
import { useSelectedModelConfig } from 'services/api/hooks/useSelectedModelConfig';

export const Prompts = memo(() => {
  const modelSupportsNegativePrompt = useAppSelector(selectModelSupportsNegativePrompt);
  const modelSupportsRefImages = useAppSelector(selectModelSupportsRefImages);
  const hasNegativePrompt = useAppSelector(selectHasNegativePrompt);
  const modelConfig = useSelectedModelConfig();

  // Qwen Image models only support ref images in the "edit" variant
  const showRefImages = useMemo(() => {
    if (!modelSupportsRefImages) {
      return false;
    }
    if (modelConfig?.base === 'qwen-image') {
      const variant = 'variant' in modelConfig ? modelConfig.variant : null;
      if (variant !== 'edit') {
        return false;
      }
    }
    return true;
  }, [modelSupportsRefImages, modelConfig]);

  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      {modelSupportsNegativePrompt && hasNegativePrompt && <ParamNegativePrompt />}
      {showRefImages && <RefImageList />}
    </Flex>
  );
});

Prompts.displayName = 'Prompts';
