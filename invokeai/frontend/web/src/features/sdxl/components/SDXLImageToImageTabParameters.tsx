import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import ParamNoiseCollapse from 'features/parameters/components/Parameters/Noise/ParamNoiseCollapse';
// import ParamVariationCollapse from 'features/parameters/components/Parameters/Variations/ParamVariationCollapse';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import ImageToImageTabCoreParameters from 'features/ui/components/tabs/ImageToImage/ImageToImageTabCoreParameters';
import ParamSDXLNegativeStyleConditioning from './ParamSDXLNegativeStyleConditioning';
import ParamSDXLPositiveStyleConditioning from './ParamSDXLPositiveStyleConditioning';

const SDXLImageToImageTabParameters = () => {
  return (
    <>
      <ParamPositiveConditioning />
      <ParamSDXLPositiveStyleConditioning />
      <ParamNegativeConditioning />
      <ParamSDXLNegativeStyleConditioning />
      <ProcessButtons />
      <ImageToImageTabCoreParameters />
      <ParamDynamicPromptsCollapse />
      <ParamNoiseCollapse />
    </>
  );
};

export default SDXLImageToImageTabParameters;
