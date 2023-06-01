import { memo, useCallback, useState } from 'react';
import { ControlNetProcessorNode } from '../store/types';
import { ImageDTO } from 'services/api';
import CannyProcessor from './processors/CannyProcessor';
import {
  CONTROLNET_PROCESSORS,
  ControlNet,
  ControlNetModel,
  ControlNetProcessor,
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
  controlNetImageChanged,
  controlNetModelChanged,
  controlNetProcessedImageChanged,
  controlNetProcessorChanged,
  controlNetRemoved,
  controlNetToggled,
  controlNetWeightChanged,
  isControlNetImageProcessedToggled,
} from '../store/controlNetSlice';
import { useAppDispatch } from 'app/store/storeHooks';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import IAISlider from 'common/components/IAISlider';
import ParamControlNetIsEnabled from './parameters/ParamControlNetIsEnabled';
import ParamControlNetModel from './parameters/ParamControlNetModel';
import ParamControlNetWeight from './parameters/ParamControlNetWeight';
import ParamControlNetBeginStepPct from './parameters/ParamControlNetBeginStepPct';
import ParamControlNetEndStepPct from './parameters/ParamControlNetEndStepPct';
import {
  Box,
  Flex,
  HStack,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  VStack,
  useDisclosure,
} from '@chakra-ui/react';
import IAISelectableImage from './parameters/IAISelectableImage';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAISwitch from 'common/components/IAISwitch';
import ParamControlNetIsPreprocessed from './parameters/ParamControlNetIsPreprocessed';
import IAICollapse from 'common/components/IAICollapse';
import ControlNetProcessorCollapse from './ControlNetProcessorCollapse';
import IAICustomSelect from 'common/components/IAICustomSelect';

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
    processor,
  } = props.controlNet;
  const dispatch = useAppDispatch();

  const handleProcessorTypeChanged = useCallback(
    (processor: string | null | undefined) => {
      dispatch(
        controlNetProcessorChanged({
          controlNetId,
          processor: processor as ControlNetProcessor,
        })
      );
    },
    [controlNetId, dispatch]
  );

  const handleControlImageChanged = useCallback(
    (controlImage: ImageDTO) => {
      dispatch(controlNetImageChanged({ controlNetId, controlImage }));
    },
    [controlNetId, dispatch]
  );

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
            <IAICustomSelect
              label="Processor"
              items={CONTROLNET_PROCESSORS}
              selectedItem={processor}
              setSelectedItem={handleProcessorTypeChanged}
            />
            <ProcessorComponent
              controlNetId={controlNetId}
              controlImage={controlImage}
              processedControlImage={processedControlImage}
              type={processor}
            />
          </TabPanel>
        </TabPanels>
      </Tabs>
      <IAIButton onClick={handleControlNetRemoved}>Remove ControlNet</IAIButton>
    </Flex>
  );
};

export default memo(ControlNet);

export type ControlNetProcessorProps = {
  controlNetId: string;
  controlImage: ImageDTO | null;
  processedControlImage: ImageDTO | null;
  type: ControlNetProcessor;
};

const ProcessorComponent = (props: ControlNetProcessorProps) => {
  const { type } = props;
  if (type === 'canny') {
    return <CannyProcessor {...props} />;
  }
  return null;
};
