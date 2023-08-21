import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamControlNetCollapse from 'features/parameters/components/Parameters/ControlNet/ParamControlNetCollapse';
import ParamNoiseCollapse from 'features/parameters/components/Parameters/Noise/ParamNoiseCollapse';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import ParamSDXLPromptArea from './ParamSDXLPromptArea';
import ParamSDXLRefinerCollapse from './ParamSDXLRefinerCollapse';
import SDXLImageToImageTabCoreParameters from './SDXLImageToImageTabCoreParameters';
import { memo } from 'react';

const SDXLImageToImageTabParameters = () => {
  return (
    <>
      <ParamSDXLPromptArea />
      <ProcessButtons />
      <SDXLImageToImageTabCoreParameters />
      <ParamSDXLRefinerCollapse />
      <ParamControlNetCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamNoiseCollapse />
    </>
  );
};

export default memo(SDXLImageToImageTabParameters);
