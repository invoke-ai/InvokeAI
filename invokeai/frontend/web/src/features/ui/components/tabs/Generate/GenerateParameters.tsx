import {
  AspectRatio,
  Box,
  Flex,
  Select,
  Slider,
  SliderFilledTrack,
  SliderThumb,
  SliderTrack,
  Text,
} from '@chakra-ui/react';
import { Feature } from 'app/features';
import IAISlider from 'common/components/IAISlider';
import IAISwitch from 'common/components/IAISwitch';
import ImageToImageSettings from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageSettings';
import ImageToImageToggle from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageToggle';
import OutputSettings from 'features/parameters/components/AdvancedParameters/Output/OutputSettings';
import SymmetrySettings from 'features/parameters/components/AdvancedParameters/Output/SymmetrySettings';
import SymmetryToggle from 'features/parameters/components/AdvancedParameters/Output/SymmetryToggle';
import RandomizeSeed from 'features/parameters/components/AdvancedParameters/Seed/RandomizeSeed';
import SeedSettings from 'features/parameters/components/AdvancedParameters/Seed/SeedSettings';
import GenerateVariationsToggle from 'features/parameters/components/AdvancedParameters/Variations/GenerateVariations';
import VariationsSettings from 'features/parameters/components/AdvancedParameters/Variations/VariationsSettings';
import DimensionsSettings from 'features/parameters/components/ImageDimensions/DimensionsSettings';
import MainSettings from 'features/parameters/components/MainParameters/MainSettings';
import ParametersAccordion, {
  ParametersAccordionItems,
} from 'features/parameters/components/ParametersAccordion';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import NegativePromptInput from 'features/parameters/components/PromptInput/NegativePromptInput';
import PromptInput from 'features/parameters/components/PromptInput/PromptInput';
import { findIndex } from 'lodash';
import { memo, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';

const GenerateParameters = () => {
  const { t } = useTranslation();

  const generateAccordionItems: ParametersAccordionItems = useMemo(
    () => ({
      // general: {
      //   name: 'general',
      //   header: `${t('parameters.general')}`,
      //   feature: undefined,
      //   content: <MainSettings />,
      // },
      seed: {
        name: 'seed',
        header: `${t('parameters.seed')}`,
        feature: Feature.SEED,
        content: <SeedSettings />,
        additionalHeaderComponents: <RandomizeSeed />,
      },
      // imageToImage: {
      //   name: 'imageToImage',
      //   header: `${t('parameters.imageToImage')}`,
      //   feature: undefined,
      //   content: <ImageToImageSettings />,
      //   additionalHeaderComponents: <ImageToImageToggle />,
      // },
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
    }),
    [t]
  );

  return (
    <Flex flexDir="column" gap={2}>
      <PromptInput />
      <NegativePromptInput />
      <ProcessButtons />
      <Flex
        sx={{
          flexDirection: 'column',
          gap: 2,
          bg: 'base.800',
          p: 4,
          pb: 6,
          borderRadius: 'base',
        }}
      >
        <MainSettings />
      </Flex>
      <ImageToImageToggle />
      <ParametersAccordion accordionItems={generateAccordionItems} />
    </Flex>
  );
};

export default memo(GenerateParameters);
