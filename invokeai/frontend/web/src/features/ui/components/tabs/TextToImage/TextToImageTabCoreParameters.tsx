import ParamIterations from 'features/parameters/components/Parameters/Core/ParamIterations';
import ParamSteps from 'features/parameters/components/Parameters/Core/ParamSteps';
import ParamCFGScale from 'features/parameters/components/Parameters/Core/ParamCFGScale';
import ParamWidth from 'features/parameters/components/Parameters/Core/ParamWidth';
import ParamHeight from 'features/parameters/components/Parameters/Core/ParamHeight';
import { Box, Flex, useDisclosure } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { memo } from 'react';
import ParamSchedulerAndModel from 'features/parameters/components/Parameters/Core/ParamSchedulerAndModel';
import IAICollapse from 'common/components/IAICollapse';
import ParamSeedFull from 'features/parameters/components/Parameters/Seed/ParamSeedFull';

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
            <ParamWidth />
            <ParamHeight />
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
            <ParamWidth />
            <ParamHeight />
          </>
        )}
      </Flex>
    </IAICollapse>
  );
};

export default memo(TextToImageTabCoreParameters);
