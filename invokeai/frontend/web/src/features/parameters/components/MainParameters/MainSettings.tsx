import { memo } from 'react';
import { Box, Flex, VStack } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';

import ModelSelect from 'features/system/components/ModelSelect';
import ParamHeight from 'features/parameters/components/Parameters/ParamHeight';
import ParamCFGScale from 'features/parameters/components/Parameters/ParamCFGScale';
import ParamIterations from 'features/parameters/components/Parameters/ParamIterations';
import ParamScheduler from 'features/parameters/components/Parameters/ParamScheduler';
import ParamSteps from 'features/parameters/components/Parameters/ParamSteps';
import ParamWidth from 'features/parameters/components/Parameters/ParamWidth';

const MainSettings = () => {
  const shouldUseSliders = useAppSelector(
    (state: RootState) => state.ui.shouldUseSliders
  );

  return shouldUseSliders ? (
    <VStack gap={2}>
      <ParamIterations />
      <ParamSteps />
      <ParamCFGScale />
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
    </VStack>
  ) : (
    <Flex gap={3} flexDirection="column">
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
    </Flex>
  );
};

export default memo(MainSettings);
