import ParamIterations from 'features/parameters/components/Parameters/Core/ParamIterations';
import ParamSteps from 'features/parameters/components/Parameters/Core/ParamSteps';
import ParamCFGScale from 'features/parameters/components/Parameters/Core/ParamCFGScale';
import ParamWidth from 'features/parameters/components/Parameters/Core/ParamWidth';
import ParamHeight from 'features/parameters/components/Parameters/Core/ParamHeight';
import ParamSampler from 'features/parameters/components/Parameters/Core/ParamSampler';
import ModelSelect from 'features/system/components/ModelSelect';
import { Box, Flex } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { memo } from 'react';

const selector = createSelector(
  uiSelector,
  (ui) => {
    const { shouldUseSliders } = ui;

    return { shouldUseSliders };
  },
  defaultSelectorOptions
);

const TextToImageTabCoreParameters = () => {
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
        </Flex>
      )}
    </Flex>
  );
};

export default memo(TextToImageTabCoreParameters);
