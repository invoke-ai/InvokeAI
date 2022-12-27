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
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasPanel() {
  const { t } = useTranslation();

  const unifiedCanvasAccordions = {
    boundingBox: {
      header: `${t('options:boundingBoxHeader')}`,
      feature: Feature.BOUNDING_BOX,
      content: <BoundingBoxSettings />,
    },
    seamCorrection: {
      header: `${t('options:seamCorrectionHeader')}`,
      feature: Feature.SEAM_CORRECTION,
      content: <SeamCorrectionOptions />,
    },
    infillAndScaling: {
      header: `${t('options:infillScalingHeader')}`,
      feature: Feature.INFILL_AND_SCALING,
      content: <InfillAndScalingOptions />,
    },
    seed: {
      header: `${t('options:seed')}`,
      feature: Feature.SEED,
      content: <SeedOptions />,
    },
    variations: {
      header: `${t('options:variations')}`,
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
        label={t('options:img2imgStrength')}
        styleClass="main-option-block image-to-image-strength-main-option"
      />
      <OptionsAccordion accordionInfo={unifiedCanvasAccordions} />
    </InvokeOptionsPanel>
  );
}
