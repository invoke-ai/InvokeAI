import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { RefImageList } from 'features/controlLayers/components/RefImage/RefImageList';
import {
  createParamsSelector,
  selectHasNegativePrompt,
  selectModelSupportsNegativePrompt,
} from 'features/controlLayers/store/paramsSlice';
import { ParamNegativePrompt } from 'features/parameters/components/Core/ParamNegativePrompt';
import { ParamPositivePrompt } from 'features/parameters/components/Core/ParamPositivePrompt';
import { ParamSDXLNegativeStylePrompt } from 'features/sdxl/components/SDXLPrompts/ParamSDXLNegativeStylePrompt';
import { ParamSDXLPositiveStylePrompt } from 'features/sdxl/components/SDXLPrompts/ParamSDXLPositiveStylePrompt';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

const selectWithStylePrompts = createParamsSelector((params) => {
  const isSDXL = params.model?.base === 'sdxl';
  const shouldConcatPrompts = params.shouldConcatPrompts;
  return isSDXL && !shouldConcatPrompts;
});

export const Prompts = memo(() => {
  const withStylePrompts = useAppSelector(selectWithStylePrompts);
  const modelSupportsNegativePrompt = useAppSelector(selectModelSupportsNegativePrompt);
  const hasNegativePrompt = useAppSelector(selectHasNegativePrompt);
  const activeTab = useAppSelector(selectActiveTab);


  return (
    <Flex flexDir="column" gap={2}>
      <ParamPositivePrompt />
      {withStylePrompts && <ParamSDXLPositiveStylePrompt />}
      {(modelSupportsNegativePrompt && hasNegativePrompt) && <ParamNegativePrompt />}
      {withStylePrompts && <ParamSDXLNegativeStylePrompt />}
      {activeTab !== 'video' && <RefImageList />}
    </Flex>
  );
});

Prompts.displayName = 'Prompts';
