import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamControlNetCollapse from 'features/parameters/components/Parameters/ControlNet/ParamControlNetCollapse';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import ParamHrfCollapse from 'features/parameters/components/Parameters/HighResFix/ParamHrfCollapse';
import { memo } from 'react';
import ParamPromptArea from '../../../../parameters/components/Parameters/Prompt/ParamPromptArea';
import TextToImageTabCoreParameters from './TextToImageTabCoreParameters';

const TextToImageTabParameters = () => {
  return (
    <>
      <ParamPromptArea />
      <TextToImageTabCoreParameters />
      <ParamControlNetCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamSymmetryCollapse />
      <ParamHrfCollapse />
      <ParamAdvancedCollapse />
    </>
  );
};

export default memo(TextToImageTabParameters);
