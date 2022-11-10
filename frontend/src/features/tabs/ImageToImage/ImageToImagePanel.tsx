import { Feature } from 'app/features';
import { RootState, useAppSelector } from 'app/store';
import FaceRestoreHeader from 'features/options/AdvancedOptions/FaceRestore/FaceRestoreHeader';
import FaceRestoreOptions from 'features/options/AdvancedOptions/FaceRestore/FaceRestoreOptions';
import ImageFit from 'features/options/AdvancedOptions/ImageToImage/ImageFit';
import ImageToImageStrength from 'features/options/AdvancedOptions/ImageToImage/ImageToImageStrength';
import OutputHeader from 'features/options/AdvancedOptions/Output/OutputHeader';
import OutputOptions from 'features/options/AdvancedOptions/Output/OutputOptions';
import SeedHeader from 'features/options/AdvancedOptions/Seed/SeedHeader';
import SeedOptions from 'features/options/AdvancedOptions/Seed/SeedOptions';
import UpscaleHeader from 'features/options/AdvancedOptions/Upscale/UpscaleHeader';
import UpscaleOptions from 'features/options/AdvancedOptions/Upscale/UpscaleOptions';
import VariationsHeader from 'features/options/AdvancedOptions/Variations/VariationsHeader';
import VariationsOptions from 'features/options/AdvancedOptions/Variations/VariationsOptions';
import MainAdvancedOptionsCheckbox from 'features/options/MainOptions/MainAdvancedOptionsCheckbox';
import MainOptions from 'features/options/MainOptions/MainOptions';
import OptionsAccordion from 'features/options/OptionsAccordion';
import ProcessButtons from 'features/options/ProcessButtons/ProcessButtons';
import PromptInput from 'features/options/PromptInput/PromptInput';
import InvokeOptionsPanel from 'features/tabs/InvokeOptionsPanel';

export default function ImageToImagePanel() {
  const showAdvancedOptions = useAppSelector(
    (state: RootState) => state.options.showAdvancedOptions
  );

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
      <MainAdvancedOptionsCheckbox />
      {showAdvancedOptions ? (
        <OptionsAccordion accordionInfo={imageToImageAccordions} />
      ) : null}
    </InvokeOptionsPanel>
  );
}
