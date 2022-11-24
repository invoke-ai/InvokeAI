// import { Feature } from 'app/features';
import { Feature } from 'app/features';
import ImageToImageStrength from 'features/options/components/AdvancedOptions/ImageToImage/ImageToImageStrength';
import SeamCorrectionOptions from 'features/options/components/AdvancedOptions/Canvas/SeamCorrectionOptions';
import SeedOptions from 'features/options/components/AdvancedOptions/Seed/SeedOptions';
import GenerateVariationsToggle from 'features/options/components/AdvancedOptions/Variations/GenerateVariations';
import VariationsOptions from 'features/options/components/AdvancedOptions/Variations/VariationsOptions';
import MainOptions from 'features/options/components/MainOptions/MainOptions';
import OptionsAccordion from 'features/options/components/OptionsAccordion';
import ProcessButtons from 'features/options/components/ProcessButtons/ProcessButtons';
import PromptInput from 'features/options/components/PromptInput/PromptInput';
import InvokeOptionsPanel from 'features/tabs/components/InvokeOptionsPanel';
import BoundingBoxSettings from 'features/options/components/AdvancedOptions/Canvas/BoundingBoxSettings/BoundingBoxSettings';
import InfillAndScalingOptions from 'features/options/components/AdvancedOptions/Canvas/InfillAndScalingOptions';

export default function UnifiedCanvasPanel() {
  const unifiedCanvasAccordions = {
    boundingBox: {
      header: 'Bounding Box',
      feature: Feature.BOUNDING_BOX,
      content: <BoundingBoxSettings />,
    },
    seamCorrection: {
      header: 'Seam Correction',
      feature: Feature.SEAM_CORRECTION,
      content: <SeamCorrectionOptions />,
    },
    infillAndScaling: {
      header: 'Infill and Scaling',
      feature: Feature.INFILL_AND_SCALING,
      content: <InfillAndScalingOptions />,
    },
    seed: {
      header: 'Seed',
      feature: Feature.SEED,
      content: <SeedOptions />,
    },
    variations: {
      header: 'Variations',
      feature: Feature.VARIATIONS,
      content: <VariationsOptions />,
      additionalHeaderComponents: <GenerateVariationsToggle />,
    },
  };

  return (
    <InvokeOptionsPanel>
      <PromptInput />
      <ProcessButtons />
      <MainOptions />
      <ImageToImageStrength
        label="Image To Image Strength"
        styleClass="main-option-block image-to-image-strength-main-option"
      />
      <OptionsAccordion accordionInfo={unifiedCanvasAccordions} />
    </InvokeOptionsPanel>
  );
}
