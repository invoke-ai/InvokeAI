import {
  Flex,
  FormControl,
  FormLabel,
  Heading,
  Spacer,
  Switch,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { ControlNetConfig } from 'features/controlNet/store/controlNetSlice';
import { ChangeEvent, memo, useCallback } from 'react';
import { controlNetToggled } from '../store/batchSlice';

type Props = {
  controlNet: ControlNetConfig;
};

const selector = createSelector(
  [stateSelector, (state, controlNetId: string) => controlNetId],
  (state, controlNetId) => {
    const isControlNetEnabled = state.batch.controlNets.includes(controlNetId);
    return { isControlNetEnabled };
  },
  defaultSelectorOptions
);

const BatchControlNet = (props: Props) => {
  const dispatch = useAppDispatch();
  const { isControlNetEnabled } = useAppSelector((state) =>
    selector(state, props.controlNet.controlNetId)
  );
  const { processorType, model } = props.controlNet;

  const handleChangeAsControlNet = useCallback(() => {
    dispatch(controlNetToggled(props.controlNet.controlNetId));
  }, [dispatch, props.controlNet.controlNetId]);

  return (
    <Flex
      layerStyle="second"
      sx={{ flexDir: 'column', gap: 1, p: 4, borderRadius: 'base' }}
    >
      <Flex sx={{ justifyContent: 'space-between' }}>
        <FormControl as={Flex} onClick={handleChangeAsControlNet}>
          <FormLabel>
            <Heading size="sm">ControlNet</Heading>
          </FormLabel>
          <Spacer />
          <Switch isChecked={isControlNetEnabled} />
        </FormControl>
      </Flex>
      <Text>
        <strong>Model:</strong> {model}
      </Text>
      <Text>
        <strong>Processor:</strong> {processorType}
      </Text>
    </Flex>
  );
};

export default memo(BatchControlNet);
