import { Feature } from 'app/features';
import FaceRestoreHeader from 'features/options/components/AdvancedOptions/FaceRestore/FaceRestoreHeader';
import FaceRestoreOptions from 'features/options/components/AdvancedOptions/FaceRestore/FaceRestoreOptions';
import ImageFit from 'features/options/components/AdvancedOptions/ImageToImage/ImageFit';
import ImageToImageStrength from 'features/options/components/AdvancedOptions/ImageToImage/ImageToImageStrength';
import OutputHeader from 'features/options/components/AdvancedOptions/Output/OutputHeader';
import OutputOptions from 'features/options/components/AdvancedOptions/Output/OutputOptions';
import SeedHeader from 'features/options/components/AdvancedOptions/Seed/SeedHeader';
import SeedOptions from 'features/options/components/AdvancedOptions/Seed/SeedOptions';
import UpscaleHeader from 'features/options/components/AdvancedOptions/Upscale/UpscaleHeader';
import UpscaleOptions from 'features/options/components/AdvancedOptions/Upscale/UpscaleOptions';
import VariationsHeader from 'features/options/components/AdvancedOptions/Variations/VariationsHeader';
import VariationsOptions from 'features/options/components/AdvancedOptions/Variations/VariationsOptions';
import MainOptions from 'features/options/components/MainOptions/MainOptions';
import OptionsAccordion from 'features/options/components/OptionsAccordion';
import ProcessButtons from 'features/options/components/ProcessButtons/ProcessButtons';
import PromptInput from 'features/options/components/PromptInput/PromptInput';
import InvokeOptionsPanel from 'features/tabs/components/InvokeOptionsPanel';

export default function ImageToImagePanel() {
  const imageToImageAccordions = {
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
    face_restore: {
      header: <FaceRestoreHeader />,
      feature: Feature.FACE_CORRECTION,
      options: <FaceRestoreOptions />,
    },
    upscale: {
      header: <UpscaleHeader />,
      feature: Feature.UPSCALE,
      options: <UpscaleOptions />,
    },
    other: {
      header: <OutputHeader />,
      feature: Feature.OTHER,
      options: <OutputOptions />,
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
      <ImageFit />
      <OptionsAccordion accordionInfo={imageToImageAccordions} />
    </InvokeOptionsPanel>
  );
}
