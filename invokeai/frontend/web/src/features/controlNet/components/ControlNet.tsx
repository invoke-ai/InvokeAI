import { memo, useCallback } from 'react';
import { RequiredControlNetProcessorNode } from '../store/types';
import { ImageDTO } from 'services/api';
import CannyProcessor from './processors/CannyProcessor';
import {
  ControlNet,
  controlNetImageChanged,
  controlNetProcessedImageChanged,
  controlNetRemoved,
} from '../store/controlNetSlice';
import { useAppDispatch } from 'app/store/storeHooks';
import ParamControlNetModel from './parameters/ParamControlNetModel';
import ParamControlNetWeight from './parameters/ParamControlNetWeight';
import ParamControlNetBeginStepPct from './parameters/ParamControlNetBeginStepPct';
import ParamControlNetEndStepPct from './parameters/ParamControlNetEndStepPct';
import {
  Flex,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@chakra-ui/react';
import IAISelectableImage from './parameters/IAISelectableImage';
import IAIButton from 'common/components/IAIButton';
import { controlNetImageProcessed } from '../store/actions';
import { FaUndo } from 'react-icons/fa';
import HedProcessor from './processors/HedProcessor';
import ParamControlNetProcessorSelect from './parameters/ParamControlNetProcessorSelect';
import ProcessorComponent from './ProcessorComponent';

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

  const handleControlImageChanged = useCallback(
    (controlImage: ImageDTO) => {
      dispatch(controlNetImageChanged({ controlNetId, controlImage }));
    },
    [controlNetId, dispatch]
  );

  const handleProcess = useCallback(() => {
    dispatch(
      controlNetImageProcessed({
        controlNetId,
      })
    );
  }, [controlNetId, dispatch]);

  const handleReset = useCallback(() => {
    dispatch(
      controlNetProcessedImageChanged({
        controlNetId,
        processedControlImage: null,
      })
    );
  }, [controlNetId, dispatch]);

  const handleControlImageReset = useCallback(() => {
    dispatch(controlNetImageChanged({ controlNetId, controlImage: null }));
  }, [controlNetId, dispatch]);

  const handleControlNetRemoved = useCallback(() => {
    dispatch(controlNetRemoved(controlNetId));
  }, [controlNetId, dispatch]);

  return (
    <Flex sx={{ flexDir: 'column', gap: 3 }}>
      <IAISelectableImage
        image={processedControlImage || controlImage}
        onChange={handleControlImageChanged}
        onReset={handleControlImageReset}
        resetIconSize="sm"
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
            <ParamControlNetBeginStepPct
              controlNetId={controlNetId}
              beginStepPct={beginStepPct}
            />
            <ParamControlNetEndStepPct
              controlNetId={controlNetId}
              endStepPct={endStepPct}
            />
          </TabPanel>
          <TabPanel sx={{ p: 0 }}>
            <ParamControlNetProcessorSelect
              controlNetId={controlNetId}
              processorNode={processorNode}
            />
            <ProcessorComponent
              controlNetId={controlNetId}
              processorNode={processorNode}
            />
            <IAIButton
              size="sm"
              onClick={handleProcess}
              isDisabled={Boolean(!controlImage)}
            >
              Preprocess
            </IAIButton>
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
