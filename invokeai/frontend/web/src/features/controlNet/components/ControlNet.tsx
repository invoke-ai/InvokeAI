import { memo, useCallback } from 'react';
import {
  ControlNet,
  controlNetProcessedImageChanged,
  controlNetRemoved,
} from '../store/controlNetSlice';
import { useAppDispatch } from 'app/store/storeHooks';
import ParamControlNetModel from './parameters/ParamControlNetModel';
import ParamControlNetWeight from './parameters/ParamControlNetWeight';
import {
  Box,
  Flex,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Text,
} from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import { FaUndo } from 'react-icons/fa';
import ParamControlNetProcessorSelect from './parameters/ParamControlNetProcessorSelect';
import ControlNetProcessorComponent from './ControlNetProcessorComponent';
import ControlNetPreprocessButton from './ControlNetPreprocessButton';
import ParamControlNetBeginEnd from './parameters/ParamControlNetBeginEnd';
import ControlNetImagePreview from './ControlNetImagePreview';

type ControlNetProps = {
  controlNet: ControlNet;
};

const ControlNet = (props: ControlNetProps) => {
  const {
    controlNetId,
    isEnabled,
    model,
    weight,
    beginStepPct,
    endStepPct,
    controlImage,
    isControlImageProcessed,
    processedControlImage,
    processorNode,
  } = props.controlNet;
  const dispatch = useAppDispatch();
  const handleReset = useCallback(() => {
    dispatch(
      controlNetProcessedImageChanged({
        controlNetId,
        processedControlImage: null,
      })
    );
  }, [controlNetId, dispatch]);

  const handleControlNetRemoved = useCallback(() => {
    dispatch(controlNetRemoved(controlNetId));
  }, [controlNetId, dispatch]);

  return (
    <Flex sx={{ flexDir: 'column', gap: 3 }}>
      <ControlNetImagePreview
        controlNetId={controlNetId}
        controlImage={controlImage}
        processedControlImage={processedControlImage}
      />
      <ParamControlNetModel controlNetId={controlNetId} model={model} />
      <Tabs
        isFitted
        orientation="horizontal"
        variant="enclosed"
        size="sm"
        colorScheme="accent"
      >
        <TabList>
          <Tab
            sx={{ 'button&': { _selected: { borderBottomColor: 'base.800' } } }}
          >
            Model Config
          </Tab>
          <Tab
            sx={{ 'button&': { _selected: { borderBottomColor: 'base.800' } } }}
          >
            Preprocess
          </Tab>
        </TabList>
        <TabPanels sx={{ pt: 2 }}>
          <TabPanel sx={{ p: 0 }}>
            <ParamControlNetWeight
              controlNetId={controlNetId}
              weight={weight}
            />
            <ParamControlNetBeginEnd
              controlNetId={controlNetId}
              beginStepPct={beginStepPct}
              endStepPct={endStepPct}
            />
          </TabPanel>
          <TabPanel sx={{ p: 0 }}>
            <ParamControlNetProcessorSelect
              controlNetId={controlNetId}
              processorNode={processorNode}
            />
            <ControlNetProcessorComponent
              controlNetId={controlNetId}
              processorNode={processorNode}
            />
            <ControlNetPreprocessButton controlNet={props.controlNet} />
            <IAIButton
              size="sm"
              leftIcon={<FaUndo />}
              onClick={handleReset}
              isDisabled={Boolean(!processedControlImage)}
            >
              Reset Processing
            </IAIButton>
          </TabPanel>
        </TabPanels>
      </Tabs>
      <IAIButton onClick={handleControlNetRemoved}>Remove ControlNet</IAIButton>
    </Flex>
  );
};

export default memo(ControlNet);
