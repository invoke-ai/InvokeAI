import { memo, useCallback } from 'react';
import {
  ControlNet,
  controlNetAdded,
  controlNetRemoved,
  controlNetToggled,
  isControlNetImageProcessedToggled,
} from '../store/controlNetSlice';
import { useAppDispatch } from 'app/store/storeHooks';
import ParamControlNetModel from './parameters/ParamControlNetModel';
import ParamControlNetWeight from './parameters/ParamControlNetWeight';
import {
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  HStack,
} from '@chakra-ui/react';
import { FaCopy, FaTrash } from 'react-icons/fa';
import ParamControlNetBeginEnd from './parameters/ParamControlNetBeginEnd';
import ControlNetImagePreview from './ControlNetImagePreview';
import IAIIconButton from 'common/components/IAIIconButton';
import { v4 as uuidv4 } from 'uuid';

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

  const handleDelete = useCallback(() => {
    dispatch(controlNetRemoved(controlNetId));
  }, [controlNetId, dispatch]);

  const handleDuplicate = useCallback(() => {
    dispatch(
      controlNetAdded({ controlNetId: uuidv4(), controlNet: props.controlNet })
    );
  }, [dispatch, props.controlNet]);

  const handleToggleIsEnabled = useCallback(() => {
    dispatch(controlNetToggled(controlNetId));
  }, [controlNetId, dispatch]);

  const handleToggleIsPreprocessed = useCallback(() => {
    dispatch(isControlNetImageProcessedToggled(controlNetId));
  }, [controlNetId, dispatch]);

  return (
    <Flex
      sx={{
        flexDir: 'column',
        gap: 2,
      }}
    >
      <HStack>
        <ParamControlNetModel controlNetId={controlNetId} model={model} />
        <IAIIconButton
          size="sm"
          tooltip="Duplicate ControlNet"
          aria-label="Duplicate ControlNet"
          onClick={handleDuplicate}
          icon={<FaCopy />}
        />
        <IAIIconButton
          size="sm"
          tooltip="Delete ControlNet"
          aria-label="Delete ControlNet"
          colorScheme="error"
          onClick={handleDelete}
          icon={<FaTrash />}
        />
      </HStack>
      <Flex
        sx={{
          gap: 4,
          paddingInlineEnd: 2,
        }}
      >
        <Flex
          sx={{
            alignItems: 'center',
            justifyContent: 'center',
            h: 32,
            w: 32,
            aspectRatio: '1/1',
          }}
        >
          <ControlNetImagePreview controlNet={props.controlNet} />
        </Flex>
        <Flex
          sx={{
            flexDir: 'column',
            gap: 2,
            w: 'full',
            justifyContent: 'space-between',
          }}
        >
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
          <Flex
            sx={{
              justifyContent: 'space-between',
            }}
          >
            <FormControl>
              <HStack>
                <Checkbox
                  isChecked={isEnabled}
                  onChange={handleToggleIsEnabled}
                />
                <FormLabel>Enabled</FormLabel>
              </HStack>
            </FormControl>
            <FormControl>
              <HStack>
                <Checkbox
                  isChecked={isControlImageProcessed}
                  onChange={handleToggleIsPreprocessed}
                />
                <FormLabel>Preprocessed</FormLabel>
              </HStack>
            </FormControl>
          </Flex>
        </Flex>
      </Flex>
    </Flex>
  );
};

export default memo(ControlNet);
