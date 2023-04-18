import { Flex } from '@chakra-ui/react';
import { Feature } from 'app/features';
import ImageToImageSettings from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageSettings';
import ImageToImageToggle from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageToggle';
import OutputSettings from 'features/parameters/components/AdvancedParameters/Output/OutputSettings';
import SymmetrySettings from 'features/parameters/components/AdvancedParameters/Output/SymmetrySettings';
import SymmetryToggle from 'features/parameters/components/AdvancedParameters/Output/SymmetryToggle';
import SeedSettings from 'features/parameters/components/AdvancedParameters/Seed/SeedSettings';
import GenerateVariationsToggle from 'features/parameters/components/AdvancedParameters/Variations/GenerateVariations';
import VariationsSettings from 'features/parameters/components/AdvancedParameters/Variations/VariationsSettings';
import MainSettings from 'features/parameters/components/MainParameters/MainSettings';
import ParametersAccordion, {
  ParametersAccordionItems,
} from 'features/parameters/components/ParametersAccordion';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import NegativePromptInput from 'features/parameters/components/PromptInput/NegativePromptInput';
import PromptInput from 'features/parameters/components/PromptInput/PromptInput';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const LinearParameters = () => {
  const { t } = useTranslation();

  const linearAccordions: ParametersAccordionItems = {
    general: {
      name: 'general',
      header: `${t('parameters.general')}`,
      feature: undefined,
      content: <MainSettings />,
    },
    seed: {
      name: 'seed',
      header: `${t('parameters.seed')}`,
      feature: Feature.SEED,
      content: <SeedSettings />,
    },
    imageToImage: {
      name: 'imageToImage',
      header: `${t('parameters.imageToImage')}`,
      feature: undefined,
      content: <ImageToImageSettings />,
      additionalHeaderComponents: <ImageToImageToggle />,
    },
    variations: {
      name: 'variations',
      header: `${t('parameters.variations')}`,
      feature: Feature.VARIATIONS,
      content: <VariationsSettings />,
      additionalHeaderComponents: <GenerateVariationsToggle />,
    },
    symmetry: {
      name: 'symmetry',
      header: `${t('parameters.symmetry')}`,
      content: <SymmetrySettings />,
      additionalHeaderComponents: <SymmetryToggle />,
    },
    other: {
      name: 'other',
      header: `${t('parameters.otherOptions')}`,
      feature: Feature.OTHER,
      content: <OutputSettings />,
    },
  };

  return (
    <Flex flexDir="column" gap={2}>
      <PromptInput />
      <NegativePromptInput />
      <ProcessButtons />
      <ParametersAccordion accordionItems={linearAccordions} />
    </Flex>
  );
};

export default memo(LinearParameters);
