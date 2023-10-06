import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamControlAdaptersCollapse from 'features/parameters/components/Parameters/ControlAdapters/ParamControlAdaptersCollapse';
import ParamPromptArea from 'features/parameters/components/Parameters/Prompt/ParamPromptArea';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import { memo } from 'react';
import ImageToImageTabCoreParameters from './ImageToImageTabCoreParameters';

const ImageToImageTabParameters = () => {
  return (
    <>
      <ParamPromptArea />
      <ImageToImageTabCoreParameters />
      <ParamControlAdaptersCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamSymmetryCollapse />
      <ParamAdvancedCollapse />
    </>
  );
};

export default memo(ImageToImageTabParameters);
