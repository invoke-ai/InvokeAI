import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamControlAdaptersCollapse from 'features/parameters/components/Parameters/ControlAdapters/ParamControlAdaptersCollapse';
import { memo } from 'react';
import ParamSDXLPromptArea from './ParamSDXLPromptArea';
import ParamSDXLRefinerCollapse from './ParamSDXLRefinerCollapse';
import SDXLImageToImageTabCoreParameters from './SDXLImageToImageTabCoreParameters';

const SDXLImageToImageTabParameters = () => {
  return (
    <>
      <ParamSDXLPromptArea />
      <SDXLImageToImageTabCoreParameters />
      <ParamSDXLRefinerCollapse />
      <ParamControlAdaptersCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamAdvancedCollapse />
    </>
  );
};

export default memo(SDXLImageToImageTabParameters);
