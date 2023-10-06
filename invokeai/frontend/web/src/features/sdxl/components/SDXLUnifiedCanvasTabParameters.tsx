import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamCompositingSettingsCollapse from 'features/parameters/components/Parameters/Canvas/Compositing/ParamCompositingSettingsCollapse';
import ParamInfillAndScalingCollapse from 'features/parameters/components/Parameters/Canvas/InfillAndScaling/ParamInfillAndScalingCollapse';
import ParamControlAdaptersCollapse from 'features/parameters/components/Parameters/ControlAdapters/ParamControlAdaptersCollapse';
import ParamSDXLPromptArea from './ParamSDXLPromptArea';
import ParamSDXLRefinerCollapse from './ParamSDXLRefinerCollapse';
import SDXLUnifiedCanvasTabCoreParameters from './SDXLUnifiedCanvasTabCoreParameters';

export default function SDXLUnifiedCanvasTabParameters() {
  return (
    <>
      <ParamSDXLPromptArea />
      <SDXLUnifiedCanvasTabCoreParameters />
      <ParamSDXLRefinerCollapse />
      <ParamControlAdaptersCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamInfillAndScalingCollapse />
      <ParamCompositingSettingsCollapse />
      <ParamAdvancedCollapse />
    </>
  );
}
