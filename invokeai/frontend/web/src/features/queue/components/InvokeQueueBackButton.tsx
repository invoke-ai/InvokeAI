import { Button, Flex, Spacer, useShiftModifier } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectDynamicPromptsIsLoading } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { QueueIterationsNumberInput } from 'features/queue/components/QueueIterationsNumberInput';
import { useInvoke } from 'features/queue/hooks/useInvoke';
import { memo } from 'react';
import { PiLightningFill, PiSparkleFill } from 'react-icons/pi';

import { QueueButtonTooltip } from './QueueButtonTooltip';

const invoke = 'Invoke';

export const InvokeButton = memo(() => {
  const queue = useInvoke();
  const shift = useShiftModifier();
  const isLoadingDynamicPrompts = useAppSelector(selectDynamicPromptsIsLoading);

  return (
    <Flex pos="relative" w="200px">
      <QueueIterationsNumberInput />
      <QueueButtonTooltip prepend={shift}>
        <Button
          onClick={shift ? queue.queueFront : queue.queueBack}
          isLoading={queue.isLoading || isLoadingDynamicPrompts}
          loadingText={invoke}
          isDisabled={queue.isDisabled}
          rightIcon={shift ? <PiLightningFill /> : <PiSparkleFill />}
          variant="solid"
          colorScheme="invokeYellow"
          size="lg"
          w="calc(100% - 60px)"
          flexShrink={0}
          justifyContent="space-between"
          spinnerPlacement="end"
        >
          <span>{invoke}</span>
          <Spacer />
        </Button>
      </QueueButtonTooltip>
    </Flex>
  );
});

InvokeButton.displayName = 'InvokeQueueBackButton';
