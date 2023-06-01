import { memo } from 'react';
import { Box, Flex, useDisclosure } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import ParamIterations from 'features/parameters/components/Parameters/Core/ParamIterations';
import ParamSteps from 'features/parameters/components/Parameters/Core/ParamSteps';
import ParamCFGScale from 'features/parameters/components/Parameters/Core/ParamCFGScale';
import ImageToImageStrength from 'features/parameters/components/Parameters/ImageToImage/ImageToImageStrength';
import ParamSchedulerAndModel from 'features/parameters/components/Parameters/Core/ParamSchedulerAndModel';
import ParamBoundingBoxWidth from 'features/parameters/components/Parameters/Canvas/BoundingBox/ParamBoundingBoxWidth';
import ParamBoundingBoxHeight from 'features/parameters/components/Parameters/Canvas/BoundingBox/ParamBoundingBoxHeight';
import ParamSeedFull from 'features/parameters/components/Parameters/Seed/ParamSeedFull';
import IAICollapse from 'common/components/IAICollapse';

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
            <ParamSchedulerAndModel />
            <Box pt={2}>
              <ParamSeedFull />
            </Box>
            <ParamIterations />
            <ParamSteps />
            <ParamCFGScale />
            <ParamBoundingBoxWidth />
            <ParamBoundingBoxHeight />
          </>
        ) : (
          <>
            <Flex gap={3}>
              <ParamIterations />
              <ParamSteps />
              <ParamCFGScale />
            </Flex>
            <ParamSchedulerAndModel />
            <Box pt={2}>
              <ParamSeedFull />
            </Box>
            <ParamBoundingBoxWidth />
            <ParamBoundingBoxHeight />
          </>
        )}
        <ImageToImageStrength />
      </Flex>
    </IAICollapse>
  );
};

export default memo(UnifiedCanvasCoreParameters);
