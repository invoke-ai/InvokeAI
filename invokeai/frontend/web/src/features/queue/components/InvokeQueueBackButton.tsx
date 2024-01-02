import { Flex, Spacer } from '@chakra-ui/layout';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import type { InvNumberInputFieldProps } from 'common/components/InvNumberInput/types';
import { setIterations } from 'features/parameters/store/generationSlice';
import { useQueueBack } from 'features/queue/hooks/useQueueBack';
import { memo, useCallback } from 'react';
import { IoSparkles } from 'react-icons/io5';

import { QueueButtonTooltip } from './QueueButtonTooltip';

const invoke = 'Invoke';

const numberInputFieldProps: InvNumberInputFieldProps = {
  ps: 6,
  borderInlineStartRadius: 'base',
  h: 'full',
  textAlign: 'center',
  fontSize: 'md',
  fontWeight: 'semibold',
};

const selector = createMemoizedSelector([stateSelector], (state) => {
  const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
    state.config.sd.iterations;
  const { iterations } = state.generation;

  return {
    iterations,
    initial,
    min,
    sliderMax,
    inputMax,
    step: coarseStep,
    fineStep,
  };
});

export const InvokeQueueBackButton = memo(() => {
  const isLoadingDynamicPrompts = useAppSelector(
    (state) => state.dynamicPrompts.isLoading
  );
  const { queueBack, isLoading, isDisabled } = useQueueBack();
  const { iterations, step, fineStep } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const handleChange = useCallback(
    (v: number) => {
      dispatch(setIterations(v));
    },
    [dispatch]
  );

  return (
    <Flex pos="relative" flexGrow={1} minW="240px">
      <IAIInformationalPopover feature="paramIterations">
        <InvNumberInput
          step={step}
          fineStep={fineStep}
          min={1}
          max={999}
          onChange={handleChange}
          value={iterations}
          defaultValue={1}
          numberInputFieldProps={numberInputFieldProps}
          pos="absolute"
          insetInlineEnd={0}
          h="full"
          ps={0}
          w="72px"
          flexShrink={0}
          variant="darkFilled"
        />
      </IAIInformationalPopover>
      <InvButton
        onClick={queueBack}
        isLoading={isLoading || isLoadingDynamicPrompts}
        loadingText={invoke}
        isDisabled={isDisabled}
        rightIcon={<IoSparkles />}
        tooltip={<QueueButtonTooltip />}
        variant="solid"
        zIndex={1}
        colorScheme="invokeYellow"
        size="lg"
        w="calc(100% - 60px)"
        flexShrink={0}
        justifyContent="space-between"
        spinnerPlacement="end"
      >
        <span>{invoke}</span>
        <Spacer />
      </InvButton>
    </Flex>
  );
});

InvokeQueueBackButton.displayName = 'InvokeQueueBackButton';
