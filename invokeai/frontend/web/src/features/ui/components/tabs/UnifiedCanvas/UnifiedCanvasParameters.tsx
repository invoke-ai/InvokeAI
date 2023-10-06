import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamLoraCollapse from 'features/lora/components/ParamLoraCollapse';
import ParamAdvancedCollapse from 'features/parameters/components/Parameters/Advanced/ParamAdvancedCollapse';
import ParamCompositingSettingsCollapse from 'features/parameters/components/Parameters/Canvas/Compositing/ParamCompositingSettingsCollapse';
import ParamInfillAndScalingCollapse from 'features/parameters/components/Parameters/Canvas/InfillAndScaling/ParamInfillAndScalingCollapse';
import ParamControlAdaptersCollapse from 'features/parameters/components/Parameters/ControlAdapters/ParamControlAdaptersCollapse';
import ParamPromptArea from 'features/parameters/components/Parameters/Prompt/ParamPromptArea';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import { memo } from 'react';
import UnifiedCanvasCoreParameters from './UnifiedCanvasCoreParameters';

const UnifiedCanvasParameters = () => {
  return (
    <>
      <ParamPromptArea />
      <UnifiedCanvasCoreParameters />
      <ParamControlAdaptersCollapse />
      <ParamLoraCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamSymmetryCollapse />
      <ParamInfillAndScalingCollapse />
      <ParamCompositingSettingsCollapse />
      <ParamAdvancedCollapse />
    </>
  );
};

export default memo(UnifiedCanvasParameters);
