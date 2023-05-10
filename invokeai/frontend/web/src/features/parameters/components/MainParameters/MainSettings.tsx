import { Box, Flex, VStack } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ModelSelect from 'features/system/components/ModelSelect';
import { memo } from 'react';
import HeightSlider from './HeightSlider';
import MainCFGScale from './MainCFGScale';
import MainIterations from './MainIterations';
import MainSampler from './MainSampler';
import MainSteps from './MainSteps';
import WidthSlider from './WidthSlider';

const MainSettings = () => {
  const shouldUseSliders = useAppSelector(
    (state: RootState) => state.ui.shouldUseSliders
  );

  return shouldUseSliders ? (
    <VStack gap={2}>
      <MainIterations />
      <MainSteps />
      <MainCFGScale />
      <WidthSlider />
      <HeightSlider />
      <Flex gap={3} w="full">
        <Box flexGrow={2}>
          <MainSampler />
        </Box>
        <Box flexGrow={3}>
          <ModelSelect />
        </Box>
      </Flex>
    </VStack>
  ) : (
    <Flex gap={3} flexDirection="column">
      <Flex gap={3}>
        <MainIterations />
        <MainSteps />
        <MainCFGScale />
      </Flex>
      <WidthSlider />
      <HeightSlider />
      <Flex gap={3} w="full">
        <Box flexGrow={2}>
          <MainSampler />
        </Box>
        <Box flexGrow={3}>
          <ModelSelect />
        </Box>
      </Flex>
    </Flex>
  );
};

export default memo(MainSettings);
