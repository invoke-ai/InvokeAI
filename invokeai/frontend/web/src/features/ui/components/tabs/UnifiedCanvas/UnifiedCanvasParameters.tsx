import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamInfillAndScalingCollapse from 'features/parameters/components/Parameters/Canvas/InfillAndScaling/ParamInfillAndScalingCollapse';
import ParamMaskAdjustmentCollapse from 'features/parameters/components/Parameters/Canvas/MaskAdjustment/ParamMaskAdjustmentCollapse';
import ParamRefinePassCollapse from 'features/parameters/components/Parameters/Canvas/SeamPainting/ParamRefinePassCollapse';
import ParamControlNetCollapse from 'features/parameters/components/Parameters/ControlNet/ParamControlNetCollapse';
import ParamPromptArea from 'features/parameters/components/Parameters/Prompt/ParamPromptArea';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import { memo } from 'react';
import UnifiedCanvasCoreParameters from './UnifiedCanvasCoreParameters';

const UnifiedCanvasParameters = () => {
  return (
    <>
      <ParamPromptArea />
      <UnifiedCanvasCoreParameters />
      <ParamControlNetCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamSymmetryCollapse />
      <ParamMaskAdjustmentCollapse />
      <ParamInfillAndScalingCollapse />
      <ParamRefinePassCollapse />
      <ParamAdvancedCollapse />
    </>
  );
};

export default memo(UnifiedCanvasParameters);
