import { Divider, Flex, VStack } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { ModelSelect } from 'exports';
import HeightSlider from './HeightSlider';
import MainCFGScale from './MainCFGScale';
import MainHeight from './MainHeight';
import MainIterations from './MainIterations';
import MainSampler from './MainSampler';
import MainSteps from './MainSteps';
import MainWidth from './MainWidth';
import WidthSlider from './WidthSlider';

export default function MainSettings() {
  const shouldUseSliders = useAppSelector(
    (state: RootState) => state.ui.shouldUseSliders
  );

  return shouldUseSliders ? (
    <VStack gap={2}>
      <MainIterations />
      <MainSteps />
      <MainCFGScale />
      <MainWidth />
      <MainHeight />
      <MainSampler />
    </VStack>
  ) : (
    <Flex gap={3} flexDirection="column">
      <Flex gap={3}>
        <MainIterations />
        <MainSteps />
        <MainCFGScale />
      </Flex>
      <Flex gap={3}>
        <MainSampler flexGrow={2} />
        <ModelSelect flexGrow={3} />
      </Flex>
      <WidthSlider />
      <HeightSlider />
    </Flex>
  );
}
