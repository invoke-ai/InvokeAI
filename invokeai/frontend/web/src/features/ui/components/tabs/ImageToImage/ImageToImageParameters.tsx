import { Flex } from '@chakra-ui/react';
import { Feature } from 'app/features';
import FaceRestoreSettings from 'features/parameters/components/AdvancedParameters/FaceRestore/FaceRestoreSettings';
import FaceRestoreToggle from 'features/parameters/components/AdvancedParameters/FaceRestore/FaceRestoreToggle';
import ImageToImageOutputSettings from 'features/parameters/components/AdvancedParameters/Output/ImageToImageOutputSettings';
import SymmetrySettings from 'features/parameters/components/AdvancedParameters/Output/SymmetrySettings';
import SymmetryToggle from 'features/parameters/components/AdvancedParameters/Output/SymmetryToggle';
import SeedSettings from 'features/parameters/components/AdvancedParameters/Seed/SeedSettings';
import UpscaleSettings from 'features/parameters/components/AdvancedParameters/Upscale/UpscaleSettings';
import UpscaleToggle from 'features/parameters/components/AdvancedParameters/Upscale/UpscaleToggle';
import GenerateVariationsToggle from 'features/parameters/components/AdvancedParameters/Variations/GenerateVariations';
import VariationsSettings from 'features/parameters/components/AdvancedParameters/Variations/VariationsSettings';
import MainSettings from 'features/parameters/components/MainParameters/MainSettings';
import ParametersAccordion from 'features/parameters/components/ParametersAccordion';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import NegativePromptInput from 'features/parameters/components/PromptInput/NegativePromptInput';
import PromptInput from 'features/parameters/components/PromptInput/PromptInput';
import { useTranslation } from 'react-i18next';
import ImageToImageSettings from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageSettings';
import { memo } from 'react';

const ImageToImageParameters = () => {
  const { t } = useTranslation();

  const imageToImageAccordions = {
    general: {
      header: `${t('parameters.general')}`,
      feature: undefined,
      content: <MainSettings />,
    },
    imageToImage: {
      header: `${t('parameters.imageToImage')}`,
      feature: undefined,
      content: <ImageToImageSettings />,
    },
    seed: {
      header: `${t('parameters.seed')}`,
      feature: Feature.SEED,
      content: <SeedSettings />,
    },
    variations: {
      header: `${t('parameters.variations')}`,
      feature: Feature.VARIATIONS,
      content: <VariationsSettings />,
      additionalHeaderComponents: <GenerateVariationsToggle />,
    },
    face_restore: {
      header: `${t('parameters.faceRestoration')}`,
      feature: Feature.FACE_CORRECTION,
      content: <FaceRestoreSettings />,
      additionalHeaderComponents: <FaceRestoreToggle />,
    },
    upscale: {
      header: `${t('parameters.upscaling')}`,
      feature: Feature.UPSCALE,
      content: <UpscaleSettings />,
      additionalHeaderComponents: <UpscaleToggle />,
    },
    symmetry: {
      header: `${t('parameters.symmetry')}`,
      content: <SymmetrySettings />,
      additionalHeaderComponents: <SymmetryToggle />,
    },
    other: {
      header: `${t('parameters.otherOptions')}`,
      feature: Feature.OTHER,
      content: <ImageToImageOutputSettings />,
    },
  };

  return (
    <Flex flexDir="column" gap={2} position="relative">
      <PromptInput />
      <NegativePromptInput />
      <ProcessButtons />
      <ParametersAccordion accordionInfo={imageToImageAccordions} />
    </Flex>
  );
};

export default memo(ImageToImageParameters);
