import { memo } from 'react';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamVariationCollapse from 'features/parameters/components/Parameters/Variations/ParamVariationCollapse';
import ParamNoiseCollapse from 'features/parameters/components/Parameters/Noise/ParamNoiseCollapse';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import ParamSeamlessCollapse from 'features/parameters/components/Parameters/Seamless/ParamSeamlessCollapse';
import ImageToImageTabCoreParameters from './ImageToImageTabCoreParameters';
import ParamControlNetCollapse from 'features/parameters/components/Parameters/ControlNet/ParamControlNetCollapse';
import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';

const ImageToImageTabParameters = () => {
  return (
    <>
      <ParamPositiveConditioning />
      <ParamNegativeConditioning />
      <ProcessButtons />
      <ImageToImageTabCoreParameters />
      <ParamDynamicPromptsCollapse />
      <ParamControlNetCollapse />
      <ParamVariationCollapse />
      <ParamNoiseCollapse />
      <ParamSymmetryCollapse />
      <ParamSeamlessCollapse />
    </>
  );
};

export default memo(ImageToImageTabParameters);
