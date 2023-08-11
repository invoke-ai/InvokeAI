import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamInfillAndScalingCollapse from 'features/parameters/components/Parameters/Canvas/InfillAndScaling/ParamInfillAndScalingCollapse';
import ParamMaskAdjustmentCollapse from 'features/parameters/components/Parameters/Canvas/MaskAdjustment/ParamMaskAdjustmentCollapse';
import ParamControlNetCollapse from 'features/parameters/components/Parameters/ControlNet/ParamControlNetCollapse';
import ParamNoiseCollapse from 'features/parameters/components/Parameters/Noise/ParamNoiseCollapse';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import UnifiedCanvasCoreParameters from 'features/ui/components/tabs/UnifiedCanvas/UnifiedCanvasCoreParameters';
import ParamSDXLPromptArea from './ParamSDXLPromptArea';
import ParamSDXLRefinerCollapse from './ParamSDXLRefinerCollapse';

export default function SDXLUnifiedCanvasTabParameters() {
  return (
    <>
      <ParamSDXLPromptArea />
      <ProcessButtons />
      <UnifiedCanvasCoreParameters />
      <ParamSDXLRefinerCollapse />
      <ParamControlNetCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamNoiseCollapse />
      <ParamMaskAdjustmentCollapse />
      <ParamInfillAndScalingCollapse />
      <ParamAdvancedCollapse />
    </>
  );
}
