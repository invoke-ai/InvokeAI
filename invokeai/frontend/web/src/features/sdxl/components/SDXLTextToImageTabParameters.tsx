import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamNoiseCollapse from 'features/parameters/components/Parameters/Noise/ParamNoiseCollapse';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import TextToImageTabCoreParameters from 'features/ui/components/tabs/TextToImage/TextToImageTabCoreParameters';
import ParamSDXLPromptArea from './ParamSDXLPromptArea';
import ParamSDXLRefinerCollapse from './ParamSDXLRefinerCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';

const SDXLTextToImageTabParameters = () => {
  return (
    <>
      <ParamSDXLPromptArea />
      <ProcessButtons />
      <TextToImageTabCoreParameters />
      <ParamSDXLRefinerCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamNoiseCollapse />
    </>
  );
};

export default SDXLTextToImageTabParameters;
