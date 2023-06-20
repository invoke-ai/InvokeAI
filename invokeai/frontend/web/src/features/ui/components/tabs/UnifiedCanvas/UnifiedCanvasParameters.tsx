import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import ParamVariationCollapse from 'features/parameters/components/Parameters/Variations/ParamVariationCollapse';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import ParamInfillAndScalingCollapse from 'features/parameters/components/Parameters/Canvas/InfillAndScaling/ParamInfillAndScalingCollapse';
import ParamSeamCorrectionCollapse from 'features/parameters/components/Parameters/Canvas/SeamCorrection/ParamSeamCorrectionCollapse';
import UnifiedCanvasCoreParameters from './UnifiedCanvasCoreParameters';
import { memo } from 'react';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamControlNetCollapse from 'features/parameters/components/Parameters/ControlNet/ParamControlNetCollapse';

const UnifiedCanvasParameters = () => {
  return (
    <>
      <ParamPositiveConditioning />
      <ParamNegativeConditioning />
      <ProcessButtons />
      <UnifiedCanvasCoreParameters />
      <ParamControlNetCollapse />
      <ParamVariationCollapse />
      <ParamSymmetryCollapse />
      <ParamSeamCorrectionCollapse />
      <ParamInfillAndScalingCollapse />
    </>
  );
};

export default memo(UnifiedCanvasParameters);
