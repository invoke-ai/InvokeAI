import { memo } from 'react';
import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import ModelSelect from 'features/system/components/ModelSelect';
import ParamIterations from 'features/parameters/components/Parameters/Core/ParamIterations';
import ParamSteps from 'features/parameters/components/Parameters/Core/ParamSteps';
import ParamCFGScale from 'features/parameters/components/Parameters/Core/ParamCFGScale';
import ParamWidth from 'features/parameters/components/Parameters/Core/ParamWidth';
import ParamHeight from 'features/parameters/components/Parameters/Core/ParamHeight';
import ImageToImageStrength from 'features/parameters/components/Parameters/ImageToImage/ImageToImageStrength';
import ImageToImageFit from 'features/parameters/components/Parameters/ImageToImage/ImageToImageFit';
import ParamSampler from 'features/parameters/components/Parameters/Core/ParamSampler';

const selector = createSelector(
  uiSelector,
  (ui) => {
    const { shouldUseSliders } = ui;

    return { shouldUseSliders };
  },
  defaultSelectorOptions
);

const UnifiedCanvasCoreParameters = () => {
  const { shouldUseSliders } = useAppSelector(selector);

  return (
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
              <ParamSampler />
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
          <Flex gap={3} w="full">
            <Box flexGrow={2}>
              <ParamSampler />
            </Box>
            <Box flexGrow={3}>
              <ModelSelect />
            </Box>
          </Flex>
          <ParamWidth />
          <ParamHeight />
          <ImageToImageStrength />
        </Flex>
      )}
    </Flex>
  );
};

export default memo(UnifiedCanvasCoreParameters);
