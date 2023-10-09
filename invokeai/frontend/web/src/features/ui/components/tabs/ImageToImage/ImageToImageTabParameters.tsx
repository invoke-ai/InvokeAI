import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamControlNetCollapse from 'features/parameters/components/Parameters/ControlNet/ParamControlNetCollapse';
import ParamPromptArea from 'features/parameters/components/Parameters/Prompt/ParamPromptArea';
import ParamHrfCollapse from 'features/parameters/components/Parameters/HighResFix/ParamHrfCollapse';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import { memo } from 'react';
import ImageToImageTabCoreParameters from './ImageToImageTabCoreParameters';

const ImageToImageTabParameters = () => {
  return (
    <>
      <ParamPromptArea />
      <ImageToImageTabCoreParameters />
      <ParamControlNetCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamSymmetryCollapse />
      <ParamHrfCollapse />
      <ParamAdvancedCollapse />
    </>
  );
};

export default memo(ImageToImageTabParameters);
