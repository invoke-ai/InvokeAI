import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ControlAdaptersCollapse from 'features/controlAdapters/components/ControlAdaptersCollapse';
import TextToImageTabCoreParameters from 'features/ui/components/tabs/TextToImage/TextToImageTabCoreParameters';
import { memo } from 'react';
import ParamSDXLPromptArea from './ParamSDXLPromptArea';
import ParamSDXLRefinerCollapse from './ParamSDXLRefinerCollapse';

const SDXLTextToImageTabParameters = () => {
  return (
    <>
      <ParamSDXLPromptArea />
      <TextToImageTabCoreParameters />
      <ParamSDXLRefinerCollapse />
      <ControlAdaptersCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamAdvancedCollapse />
    </>
  );
};

export default memo(SDXLTextToImageTabParameters);
