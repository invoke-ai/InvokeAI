// import { Feature } from 'app/features';
import { Flex } from '@chakra-ui/react';
import { Feature } from 'app/features';
import BoundingBoxSettings from 'features/parameters/components/AdvancedParameters/Canvas/BoundingBox/BoundingBoxSettings';
import InfillAndScalingSettings from 'features/parameters/components/AdvancedParameters/Canvas/InfillAndScalingSettings';
import SeamCorrectionSettings from 'features/parameters/components/AdvancedParameters/Canvas/SeamCorrection/SeamCorrectionSettings';
import ImageToImageStrength from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageStrength';
import SymmetrySettings from 'features/parameters/components/AdvancedParameters/Output/SymmetrySettings';
import SymmetryToggle from 'features/parameters/components/AdvancedParameters/Output/SymmetryToggle';
import SeedSettings from 'features/parameters/components/AdvancedParameters/Seed/SeedSettings';
import GenerateVariationsToggle from 'features/parameters/components/AdvancedParameters/Variations/GenerateVariations';
import VariationsSettings from 'features/parameters/components/AdvancedParameters/Variations/VariationsSettings';
import MainSettings from 'features/parameters/components/MainParameters/MainParameters';
import ParametersAccordion from 'features/parameters/components/ParametersAccordion';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import NegativePromptInput from 'features/parameters/components/PromptInput/NegativePromptInput';
import PromptInput from 'features/parameters/components/PromptInput/PromptInput';
import InvokeOptionsPanel from 'features/ui/components/InvokeParametersPanel';
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasPanel() {
  const { t } = useTranslation();

  const unifiedCanvasAccordions = {
    seed: {
      header: `${t('parameters.seed')}`,
      feature: Feature.SEED,
      content: <SeedSettings />,
    },
    boundingBox: {
      header: `${t('parameters.boundingBoxHeader')}`,
      feature: Feature.BOUNDING_BOX,
      content: <BoundingBoxSettings />,
    },
    seamCorrection: {
      header: `${t('parameters.seamCorrectionHeader')}`,
      feature: Feature.SEAM_CORRECTION,
      content: <SeamCorrectionSettings />,
    },
    infillAndScaling: {
      header: `${t('parameters.infillScalingHeader')}`,
      feature: Feature.INFILL_AND_SCALING,
      content: <InfillAndScalingSettings />,
    },
    variations: {
      header: `${t('parameters.variations')}`,
      feature: Feature.VARIATIONS,
      content: <VariationsSettings />,
      additionalHeaderComponents: <GenerateVariationsToggle />,
    },
    symmetry: {
      header: `${t('parameters.symmetry')}`,
      content: <SymmetrySettings />,
      additionalHeaderComponents: <SymmetryToggle />,
    },
  };

  const unifiedCanvasImg2ImgAccordion = {
    unifiedCanvasImg2Img: {
      header: `${t('parameters.imageToImage')}`,
      feature: undefined,
      content: (
        <ImageToImageStrength
          label={t('parameters.img2imgStrength')}
          styleClass="main-settings-block image-to-image-strength-main-option"
        />
      ),
    },
  };

  return (
    <InvokeOptionsPanel>
      <Flex flexDir="column" rowGap="0.5rem">
        <PromptInput />
        <NegativePromptInput />
      </Flex>
      <ProcessButtons />
      <MainSettings />
      <ParametersAccordion accordionInfo={unifiedCanvasImg2ImgAccordion} />
      <ParametersAccordion accordionInfo={unifiedCanvasAccordions} />
    </InvokeOptionsPanel>
  );
}
