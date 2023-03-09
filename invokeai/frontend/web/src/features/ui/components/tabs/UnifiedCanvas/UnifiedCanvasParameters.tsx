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
import MainSettings from 'features/parameters/components/MainParameters/MainSettings';
import ParametersAccordion from 'features/parameters/components/ParametersAccordion';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import NegativePromptInput from 'features/parameters/components/PromptInput/NegativePromptInput';
import PromptInput from 'features/parameters/components/PromptInput/PromptInput';
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasParameters() {
  const { t } = useTranslation();

  const unifiedCanvasAccordions = {
    general: {
      header: `${t('parameters.general')}`,
      feature: undefined,
      content: <MainSettings />,
    },
    unifiedCanvasImg2Img: {
      header: `${t('parameters.imageToImage')}`,
      feature: undefined,
      content: <ImageToImageStrength label={t('parameters.img2imgStrength')} />,
    },
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

  return (
    <Flex flexDir="column" gap={2} position="relative">
      <PromptInput />
      <NegativePromptInput />
      <ProcessButtons />
      <ParametersAccordion accordionInfo={unifiedCanvasAccordions} />
    </Flex>
  );
}
