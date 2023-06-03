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
    <Flex
      sx={{
        gap: 4,
        p: 2,
        paddingInlineEnd: 4,
        bg: 'base.850',
        borderRadius: 'base',
      }}
    >
      <Flex
        sx={{
          alignItems: 'center',
          justifyContent: 'center',
          h: 36,
          w: 36,
        }}
      >
        <ControlNetImagePreview
          controlNetId={controlNetId}
          controlImage={controlImage}
          processedControlImage={processedControlImage}
        />
      </Flex>
      <Flex sx={{ flexDir: 'column', gap: 2, w: 'full', h: 'full' }}>
        <ParamControlNetModel controlNetId={controlNetId} model={model} />
        <ParamControlNetWeight
          controlNetId={controlNetId}
          weight={weight}
          mini
        />
        <ParamControlNetBeginEnd
          controlNetId={controlNetId}
          beginStepPct={beginStepPct}
          endStepPct={endStepPct}
          mini
        />
      </Flex>
    </Flex>
  );
};

export default memo(ControlNet);
