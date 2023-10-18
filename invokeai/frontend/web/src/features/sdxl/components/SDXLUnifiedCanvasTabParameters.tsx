import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamCompositingSettingsCollapse from 'features/parameters/components/Parameters/Canvas/Compositing/ParamCompositingSettingsCollapse';
import ParamInfillAndScalingCollapse from 'features/parameters/components/Parameters/Canvas/InfillAndScaling/ParamInfillAndScalingCollapse';
import ControlAdaptersCollapse from 'features/controlAdapters/components/ControlAdaptersCollapse';
import ParamSDXLPromptArea from './ParamSDXLPromptArea';
import ParamSDXLRefinerCollapse from './ParamSDXLRefinerCollapse';
import SDXLUnifiedCanvasTabCoreParameters from './SDXLUnifiedCanvasTabCoreParameters';

export default function SDXLUnifiedCanvasTabParameters() {
  return (
    <>
      <ParamSDXLPromptArea />
      <SDXLUnifiedCanvasTabCoreParameters />
      <ParamSDXLRefinerCollapse />
      <ControlAdaptersCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamInfillAndScalingCollapse />
      <ParamCompositingSettingsCollapse />
      <ParamAdvancedCollapse />
    </>
  );
}
