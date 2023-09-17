import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import ParamCFGScale from 'features/parameters/components/Parameters/Core/ParamCFGScale';
import ParamIterations from 'features/parameters/components/Parameters/Core/ParamIterations';
import ParamModelandVAEandScheduler from 'features/parameters/components/Parameters/Core/ParamModelandVAEandScheduler';
import ParamSize from 'features/parameters/components/Parameters/Core/ParamSize';
import ParamSteps from 'features/parameters/components/Parameters/Core/ParamSteps';
import ImageToImageFit from 'features/parameters/components/Parameters/ImageToImage/ImageToImageFit';
import ParamSeedFull from 'features/parameters/components/Parameters/Seed/ParamSeedFull';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import ParamSDXLImg2ImgDenoisingStrength from './ParamSDXLImg2ImgDenoisingStrength';

const selector = createSelector(
  [uiSelector, generationSelector],
  (ui, generation) => {
    const { shouldUseSliders } = ui;
    const { shouldRandomizeSeed } = generation;

    const activeLabel = !shouldRandomizeSeed ? 'Manual Seed' : undefined;

    return { shouldUseSliders, activeLabel };
  },
  defaultSelectorOptions
);

const SDXLImageToImageTabCoreParameters = () => {
  const { shouldUseSliders, activeLabel } = useAppSelector(selector);

  return (
    <IAICollapse label="General" activeLabel={activeLabel} defaultIsOpen={true}>
      <Flex
        sx={{
          flexDirection: 'column',
          gap: 3,
        }}
      >
        {shouldUseSliders ? (
          <>
            <ParamIterations />
            <ParamSteps />
            <ParamCFGScale />
            <ParamModelandVAEandScheduler />
            <Box pt={2}>
              <ParamSeedFull />
            </Box>
            <ParamSize />
          </>
        ) : (
          <>
            <Flex gap={3}>
              <ParamIterations />
              <ParamSteps />
              <ParamCFGScale />
            </Flex>
            <ParamModelandVAEandScheduler />
            <Box pt={2}>
              <ParamSeedFull />
            </Box>
            <ParamSize />
          </>
        )}
        <ParamSDXLImg2ImgDenoisingStrength />
        <ImageToImageFit />
      </Flex>
    </IAICollapse>
  );
};

export default memo(SDXLImageToImageTabCoreParameters);
