import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamNoiseCollapse from 'features/parameters/components/Parameters/Noise/ParamNoiseCollapse';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import ParamSDXLPromptArea from './ParamSDXLPromptArea';
import ParamSDXLRefinerCollapse from './ParamSDXLRefinerCollapse';
import SDXLImageToImageTabCoreParameters from './SDXLImageToImageTabCoreParameters';

const SDXLImageToImageTabParameters = () => {
  return (
    <>
      <ParamSDXLPromptArea />
      <ProcessButtons />
      <SDXLImageToImageTabCoreParameters />
      <ParamSDXLRefinerCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamNoiseCollapse />
    </>
  );
};

export default SDXLImageToImageTabParameters;
