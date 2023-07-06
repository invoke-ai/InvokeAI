import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamInfillAndScalingCollapse from 'features/parameters/components/Parameters/Canvas/InfillAndScaling/ParamInfillAndScalingCollapse';
import ParamSeamCorrectionCollapse from 'features/parameters/components/Parameters/Canvas/SeamCorrection/ParamSeamCorrectionCollapse';
import ParamControlNetCollapse from 'features/parameters/components/Parameters/ControlNet/ParamControlNetCollapse';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import ParamVariationCollapse from 'features/parameters/components/Parameters/Variations/ParamVariationCollapse';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import { memo } from 'react';
import UnifiedCanvasCoreParameters from './UnifiedCanvasCoreParameters';

const UnifiedCanvasParameters = () => {
  return (
    <>
      <ParamPositiveConditioning />
      <ParamNegativeConditioning />
      <ProcessButtons />
      <UnifiedCanvasCoreParameters />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamControlNetCollapse />
      <ParamVariationCollapse />
      <ParamSymmetryCollapse />
      <ParamSeamCorrectionCollapse />
      <ParamInfillAndScalingCollapse />
    </>
  );
};

export default memo(UnifiedCanvasParameters);
