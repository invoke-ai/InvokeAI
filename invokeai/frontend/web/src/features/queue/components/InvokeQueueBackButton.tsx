import { Button, Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectDynamicPromptsIsLoading } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { QueueIterationsNumberInput } from 'features/queue/components/QueueIterationsNumberInput';
import { useQueueBack } from 'features/queue/hooks/useQueueBack';
import { memo } from 'react';
import { RiSparkling2Fill } from 'react-icons/ri';

import { QueueButtonTooltip } from './QueueButtonTooltip';

const invoke = 'Invoke';

export const InvokeQueueBackButton = memo(() => {
  const { queueBack, isLoading, isDisabled } = useQueueBack();
  const isLoadingDynamicPrompts = useAppSelector(selectDynamicPromptsIsLoading);

  return (
    <Flex pos="relative" w="200px">
      <QueueIterationsNumberInput />
      <QueueButtonTooltip>
        <Button
          onClick={queueBack}
          isLoading={isLoading || isLoadingDynamicPrompts}
          loadingText={invoke}
          isDisabled={isDisabled}
          rightIcon={<RiSparkling2Fill />}
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

InvokeQueueBackButton.displayName = 'InvokeQueueBackButton';
