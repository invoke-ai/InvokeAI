import { Box, Flex } from '@chakra-ui/react';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import { memo } from 'react';
import OverlayScrollable from '../../common/OverlayScrollable';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/ParamPositiveConditioning';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/ParamNegativeConditioning';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import ParamIterations from 'features/parameters/components/Parameters/ParamIterations';
import ParamSteps from 'features/parameters/components/Parameters/ParamSteps';
import ParamCFGScale from 'features/parameters/components/Parameters/ParamCFGScale';
import ParamWidth from 'features/parameters/components/Parameters/ParamWidth';
import ParamHeight from 'features/parameters/components/Parameters/ParamHeight';
import ParamScheduler from 'features/parameters/components/Parameters/ParamScheduler';
import ModelSelect from 'features/system/components/ModelSelect';
import ParamSeedCollapse from 'features/parameters/components/Parameters/Seed/ParamSeedCollapse';
import ParamVariationCollapse from 'features/parameters/components/Parameters/Variations/ParamVariationCollapse';
import ParamNoiseCollapse from 'features/parameters/components/Parameters/Noise/ParamNoiseCollapse';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import ParamHiresCollapse from 'features/parameters/components/Parameters/Hires/ParamHiresCollapse';
import ParamSeamlessCollapse from 'features/parameters/components/Parameters/Seamless/ParamSeamlessCollapse';
import ImageToImageStrength from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageStrength';
import ImageToImageFit from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageFit';

const selector = createSelector(
  uiSelector,
  (ui) => {
    const { shouldUseSliders } = ui;

    return { shouldUseSliders };
  },
  defaultSelectorOptions
);

const ImageTabParameters = () => {
  const { shouldUseSliders } = useAppSelector(selector);

  return (
    <OverlayScrollable>
      <Flex
        sx={{
          gap: 2,
          flexDirection: 'column',
          h: 'full',
          w: 'full',
          position: 'absolute',
        }}
      >
        <ParamPositiveConditioning />
        <ParamNegativeConditioning />
        <ProcessButtons />
        <Flex
          sx={{
            flexDirection: 'column',
            gap: 2,
            bg: 'base.800',
            p: 4,
            borderRadius: 'base',
          }}
        >
          {shouldUseSliders ? (
            <Flex sx={{ gap: 3, flexDirection: 'column' }}>
              <ParamIterations />
              <ParamSteps />
              <ParamCFGScale />
              <ParamWidth />
              <ParamHeight />
              <ImageToImageStrength />
              <ImageToImageFit />
              <Flex gap={3} w="full">
                <Box flexGrow={2}>
                  <ParamScheduler />
                </Box>
                <Box flexGrow={3}>
                  <ModelSelect />
                </Box>
              </Flex>
            </Flex>
          ) : (
            <Flex sx={{ gap: 2, flexDirection: 'column' }}>
              <Flex gap={3}>
                <ParamIterations />
                <ParamSteps />
                <ParamCFGScale />
              </Flex>
              <ParamWidth />
              <ParamHeight />
              <Flex gap={3} w="full">
                <Box flexGrow={2}>
                  <ParamScheduler />
                </Box>
                <Box flexGrow={3}>
                  <ModelSelect />
                </Box>
              </Flex>
              <ImageToImageStrength />
              <ImageToImageFit />
            </Flex>
          )}
        </Flex>
        <ParamSeedCollapse />
        <ParamVariationCollapse />
        <ParamNoiseCollapse />
        <ParamSymmetryCollapse />
        <ParamSeamlessCollapse />
      </Flex>
    </OverlayScrollable>
  );
};

export default memo(ImageTabParameters);
