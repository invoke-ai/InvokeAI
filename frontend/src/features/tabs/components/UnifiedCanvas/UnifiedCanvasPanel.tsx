// import { Feature } from 'app/features';
import { Feature } from 'app/features';
import ImageToImageStrength from 'features/options/components/AdvancedOptions/ImageToImage/ImageToImageStrength';
import BoundingBoxSettings, {
  BoundingBoxSettingsHeader,
} from 'features/options/components/AdvancedOptions/Inpainting/BoundingBoxSettings/BoundingBoxSettings';
import OutpaintingOptions, {
  OutpaintingHeader,
} from 'features/options/components/AdvancedOptions/Inpainting/OutpaintingOptions';
import SeedHeader from 'features/options/components/AdvancedOptions/Seed/SeedHeader';
import SeedOptions from 'features/options/components/AdvancedOptions/Seed/SeedOptions';
import VariationsHeader from 'features/options/components/AdvancedOptions/Variations/VariationsHeader';
import VariationsOptions from 'features/options/components/AdvancedOptions/Variations/VariationsOptions';
import MainOptions from 'features/options/components/MainOptions/MainOptions';
import OptionsAccordion from 'features/options/components/OptionsAccordion';
import ProcessButtons from 'features/options/components/ProcessButtons/ProcessButtons';
import PromptInput from 'features/options/components/PromptInput/PromptInput';
import InvokeOptionsPanel from 'features/tabs/components/InvokeOptionsPanel';

export default function UnifiedCanvasPanel() {
  const unifiedCanvasAccordions = {
    boundingBox: {
      header: <BoundingBoxSettingsHeader />,
      feature: Feature.BOUNDING_BOX,
      options: <BoundingBoxSettings />,
    },
    outpainting: {
      header: <OutpaintingHeader />,
      feature: Feature.OUTPAINTING,
      options: <OutpaintingOptions />,
    },
    seed: {
      header: <SeedHeader />,
      feature: Feature.SEED,
      options: <SeedOptions />,
    },
    variations: {
      header: <VariationsHeader />,
      feature: Feature.VARIATIONS,
      options: <VariationsOptions />,
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
