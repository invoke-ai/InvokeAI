import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamInfillAndScalingCollapse from 'features/parameters/components/Parameters/Canvas/InfillAndScaling/ParamInfillAndScalingCollapse';
import ParamControlNetCollapse from 'features/parameters/components/Parameters/ControlNet/ParamControlNetCollapse';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
// import ParamVariationCollapse from 'features/parameters/components/Parameters/Variations/ParamVariationCollapse';
import ParamMaskAdjustmentCollapse from 'features/parameters/components/Parameters/Canvas/MaskAdjustment/ParamMaskAdjustmentCollapse';
import ParamSeamPaintingCollapse from 'features/parameters/components/Parameters/Canvas/SeamPainting/ParamSeamPaintingCollapse';
import ParamPromptArea from 'features/parameters/components/Parameters/Prompt/ParamPromptArea';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import UnifiedCanvasCoreParameters from './UnifiedCanvasCoreParameters';

const UnifiedCanvasParameters = () => {
  return (
    <>
      <ParamPromptArea />
      <ProcessButtons />
      <UnifiedCanvasCoreParameters />
      <ParamControlNetCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      {/* <ParamVariationCollapse /> */}
      <ParamSymmetryCollapse />
      <ParamMaskAdjustmentCollapse />
      <ParamInfillAndScalingCollapse />
      <ParamSeamPaintingCollapse />
      <ParamAdvancedCollapse />
    </>
  );
};

export default UnifiedCanvasParameters;
