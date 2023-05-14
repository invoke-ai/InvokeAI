import { memo } from 'react';
import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import ParamIterations from 'features/parameters/components/Parameters/Core/ParamIterations';
import ParamSteps from 'features/parameters/components/Parameters/Core/ParamSteps';
import ParamCFGScale from 'features/parameters/components/Parameters/Core/ParamCFGScale';
import ParamWidth from 'features/parameters/components/Parameters/Core/ParamWidth';
import ParamHeight from 'features/parameters/components/Parameters/Core/ParamHeight';
import ImageToImageStrength from 'features/parameters/components/Parameters/ImageToImage/ImageToImageStrength';
import ImageToImageFit from 'features/parameters/components/Parameters/ImageToImage/ImageToImageFit';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import ParamSchedulerAndModel from 'features/parameters/components/Parameters/Core/ParamSchedulerAndModel';

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
          <ParamWidth isDisabled={!shouldFitToWidthHeight} />
          <ParamHeight isDisabled={!shouldFitToWidthHeight} />
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
          <ParamWidth isDisabled={!shouldFitToWidthHeight} />
          <ParamHeight isDisabled={!shouldFitToWidthHeight} />
          <ImageToImageStrength />
          <ImageToImageFit />
        </Flex>
      )}
    </Flex>
  );
};

export default memo(ImageToImageTabCoreParameters);
