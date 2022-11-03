import { Feature } from '../../../app/features';
import { RootState, useAppSelector } from '../../../app/store';
import FaceRestoreHeader from '../../options/AdvancedOptions/FaceRestore/FaceRestoreHeader';
import FaceRestoreOptions from '../../options/AdvancedOptions/FaceRestore/FaceRestoreOptions';
import ImageFit from '../../options/AdvancedOptions/ImageToImage/ImageFit';
import ImageToImageStrength from '../../options/AdvancedOptions/ImageToImage/ImageToImageStrength';
import OutputHeader from '../../options/AdvancedOptions/Output/OutputHeader';
import OutputOptions from '../../options/AdvancedOptions/Output/OutputOptions';
import SeedHeader from '../../options/AdvancedOptions/Seed/SeedHeader';
import SeedOptions from '../../options/AdvancedOptions/Seed/SeedOptions';
import UpscaleHeader from '../../options/AdvancedOptions/Upscale/UpscaleHeader';
import UpscaleOptions from '../../options/AdvancedOptions/Upscale/UpscaleOptions';
import VariationsHeader from '../../options/AdvancedOptions/Variations/VariationsHeader';
import VariationsOptions from '../../options/AdvancedOptions/Variations/VariationsOptions';
import MainAdvancedOptionsCheckbox from '../../options/MainOptions/MainAdvancedOptionsCheckbox';
import MainOptions from '../../options/MainOptions/MainOptions';
import OptionsAccordion from '../../options/OptionsAccordion';
import ProcessButtons from '../../options/ProcessButtons/ProcessButtons';
import PromptInput from '../../options/PromptInput/PromptInput';
import InvokeOptionsPanel from '../InvokeOptionsPanel';

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
