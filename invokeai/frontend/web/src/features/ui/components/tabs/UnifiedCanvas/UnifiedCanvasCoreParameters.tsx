import { memo } from 'react';
import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import ParamIterations from 'features/parameters/components/Parameters/Core/ParamIterations';
import ParamSteps from 'features/parameters/components/Parameters/Core/ParamSteps';
import ParamCFGScale from 'features/parameters/components/Parameters/Core/ParamCFGScale';
import ImageToImageStrength from 'features/parameters/components/Parameters/ImageToImage/ImageToImageStrength';
import ImageToImageFit from 'features/parameters/components/Parameters/ImageToImage/ImageToImageFit';
import ParamSchedulerAndModel from 'features/parameters/components/Parameters/Core/ParamSchedulerAndModel';
import ParamBoundingBoxWidth from 'features/parameters/components/Parameters/Canvas/BoundingBox/ParamBoundingBoxWidth';
import ParamBoundingBoxHeight from 'features/parameters/components/Parameters/Canvas/BoundingBox/ParamBoundingBoxHeight';

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
          <ParamBoundingBoxWidth />
          <ParamBoundingBoxHeight />
          <ImageToImageStrength />
          <ImageToImageFit />
          <ParamSchedulerAndModel />
        </Flex>
      ) : (
        <Flex sx={{ gap: 2, flexDirection: 'column' }}>
          <Flex gap={3}>
            <ParamIterations />
            <ParamSteps />
            <ParamCFGScale />
          </Flex>
          <ParamSchedulerAndModel />
          <ParamBoundingBoxWidth />
          <ParamBoundingBoxHeight />
          <ImageToImageStrength />
        </Flex>
      )}
    </Flex>
  );
};

export default memo(UnifiedCanvasCoreParameters);
