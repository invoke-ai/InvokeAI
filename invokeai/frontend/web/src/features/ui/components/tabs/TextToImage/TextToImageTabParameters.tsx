import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamControlAdaptersCollapse from 'features/parameters/components/Parameters/ControlAdapters/ParamControlAdaptersCollapse';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import { memo } from 'react';
import ParamPromptArea from '../../../../parameters/components/Parameters/Prompt/ParamPromptArea';
import TextToImageTabCoreParameters from './TextToImageTabCoreParameters';

const TextToImageTabParameters = () => {
  return (
    <>
      <ParamPromptArea />
      <TextToImageTabCoreParameters />
      <ParamControlAdaptersCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamSymmetryCollapse />
      <ParamAdvancedCollapse />
    </>
  );
};

export default memo(TextToImageTabParameters);
