import { Feature } from '../../../app/features';
import { RootState, useAppSelector } from '../../../app/store';
import FaceRestoreHeader from '../../options/AdvancedOptions/FaceRestore/FaceRestoreHeader';
import FaceRestoreOptions from '../../options/AdvancedOptions/FaceRestore/FaceRestoreOptions';
import ImageToImageStrength from '../../options/AdvancedOptions/ImageToImage/ImageToImageStrength';
import BoundingBoxDimensions from '../../options/AdvancedOptions/Inpainting/BoundingBoxDimensions';
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

export default function InpaintingPanel() {
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
  };

  return (
    <div className="image-to-image-panel">
      <PromptInput />
      <ProcessButtons />
      <MainOptions />
      <BoundingBoxDimensions />
      <ImageToImageStrength
        label="Image To Image Strength"
        styleClass="main-option-block image-to-image-strength-main-option"
      />
      <MainAdvancedOptionsCheckbox />
      {showAdvancedOptions ? (
        <OptionsAccordion accordionInfo={imageToImageAccordions} />
      ) : null}
    </div>
  );
}
