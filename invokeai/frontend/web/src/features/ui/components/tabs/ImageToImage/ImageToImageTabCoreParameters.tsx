import { Box, Flex, useDisclosure } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import ParamCFGScale from 'features/parameters/components/Parameters/Core/ParamCFGScale';
import ParamHeight from 'features/parameters/components/Parameters/Core/ParamHeight';
import ParamIterations from 'features/parameters/components/Parameters/Core/ParamIterations';
import ParamModelandVAE from 'features/parameters/components/Parameters/Core/ParamModelandVAE';
import ParamScheduler from 'features/parameters/components/Parameters/Core/ParamScheduler';
import ParamSteps from 'features/parameters/components/Parameters/Core/ParamSteps';
import ParamWidth from 'features/parameters/components/Parameters/Core/ParamWidth';
import ImageToImageFit from 'features/parameters/components/Parameters/ImageToImage/ImageToImageFit';
import ImageToImageStrength from 'features/parameters/components/Parameters/ImageToImage/ImageToImageStrength';
import ParamSeedFull from 'features/parameters/components/Parameters/Seed/ParamSeedFull';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

const selector = createSelector(
  [uiSelector, generationSelector],
  (ui, generation) => {
    const { shouldUseSliders } = ui;
    const { shouldFitToWidthHeight } = generation;

    return { shouldUseSliders, shouldFitToWidthHeight };
  },
  defaultSelectorOptions
);

const ImageToImageTabCoreParameters = () => {
  const { shouldUseSliders, shouldFitToWidthHeight } = useAppSelector(selector);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });

  return (
    <IAICollapse label={'General'} isOpen={isOpen} onToggle={onToggle}>
      <Flex
        sx={{
          flexDirection: 'column',
          gap: 3,
        }}
      >
        {shouldUseSliders ? (
          <>
            <ParamModelandVAE />
            <Box pt={2}>
              <ParamSeedFull />
            </Box>
            <ParamIterations />
            <ParamSteps />
            <ParamCFGScale />
            <ParamWidth isDisabled={!shouldFitToWidthHeight} />
            <ParamHeight isDisabled={!shouldFitToWidthHeight} />
          </>
        ) : (
          <>
            <Flex gap={3}>
              <ParamIterations />
              <ParamSteps />
              <ParamCFGScale />
            </Flex>
            <ParamModelandVAE />
            <ParamScheduler />
            <Box pt={2}>
              <ParamSeedFull />
            </Box>
            <ParamWidth isDisabled={!shouldFitToWidthHeight} />
            <ParamHeight isDisabled={!shouldFitToWidthHeight} />
          </>
        )}
        <ImageToImageStrength />
        <ImageToImageFit />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ImageToImageTabCoreParameters);
