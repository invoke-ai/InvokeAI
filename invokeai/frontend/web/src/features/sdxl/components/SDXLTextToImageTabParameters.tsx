import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import ParamNoiseCollapse from 'features/parameters/components/Parameters/Noise/ParamNoiseCollapse';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import TextToImageTabCoreParameters from 'features/ui/components/tabs/TextToImage/TextToImageTabCoreParameters';
import ParamSDXLConcatPrompt from './ParamSDXLConcatPrompt';
import ParamSDXLNegativeStyleConditioning from './ParamSDXLNegativeStyleConditioning';
import ParamSDXLPositiveStyleConditioning from './ParamSDXLPositiveStyleConditioning';
import ParamSDXLRefinerCollapse from './ParamSDXLRefinerCollapse';

const SDXLTextToImageTabParameters = () => {
  return (
    <>
      <ParamPositiveConditioning />
      <ParamSDXLPositiveStyleConditioning />
      <ParamNegativeConditioning />
      <ParamSDXLNegativeStyleConditioning />
      <ParamSDXLConcatPrompt />
      <ProcessButtons />
      <TextToImageTabCoreParameters />
      <ParamSDXLRefinerCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamNoiseCollapse />
    </>
  );
};

export default SDXLTextToImageTabParameters;
