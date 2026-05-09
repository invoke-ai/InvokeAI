import { Button, Flex, Spacer, useShiftModifier } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectDynamicPromptsIsLoading } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { QueueIterationsNumberInput } from 'features/queue/components/QueueIterationsNumberInput';
import { useInvoke } from 'features/queue/hooks/useInvoke';
import { memo } from 'react';
import { PiLightningFill, PiSparkleFill } from 'react-icons/pi';
import { useAutoAddBoard } from 'services/api/hooks/useAutoAddBoard';
import { useBoardAccess } from 'services/api/hooks/useBoardAccess';

import { InvokeButtonTooltip } from './InvokeButtonTooltip/InvokeButtonTooltip';

const invoke = 'Invoke';

export const InvokeButton = memo(() => {
  const queue = useInvoke();
  const shift = useShiftModifier();
  const isLoadingDynamicPrompts = useAppSelector(selectDynamicPromptsIsLoading);
  const autoAddBoard = useAutoAddBoard();
  const { canWriteImages } = useBoardAccess(autoAddBoard);

  return (
    <Flex pos="relative" w="200px">
      <QueueIterationsNumberInput />
      <InvokeButtonTooltip prepend={shift}>
        <Button
          onClick={shift ? queue.enqueueFront : queue.enqueueBack}
          isLoading={queue.isLoading || isLoadingDynamicPrompts}
          loadingText={invoke}
          isDisabled={queue.isDisabled || !canWriteImages}
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
      </InvokeButtonTooltip>
    </Flex>
  );
});

InvokeButton.displayName = 'InvokeQueueBackButton';
