import type { FlexProps, ProgressProps } from '@invoke-ai/ui-library';
import { Flex, Progress } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { $activeProgressEvents, $isConnected, $loadingModelsCount } from 'services/events/stores';

type ProgressBarProps = ProgressProps & {
  /** Applied to the Flex that stacks the per-session bars. Use for positioning (e.g. absolute). */
  containerProps?: FlexProps;
};

type BarDescriptor = {
  key: number | string;
  value: number;
  isIndeterminate: boolean;
};

const ProgressBar = ({ containerProps, ...props }: ProgressBarProps) => {
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const isConnected = useStore($isConnected);
  const activeProgressEvents = useStore($activeProgressEvents);
  const loadingModelsCount = useStore($loadingModelsCount);

  const bars = useMemo<BarDescriptor[]>(() => {
    // One bar per in-flight session (multi-GPU). Each session's progress is tracked independently, so
    // the bars no longer jump back and forth when several sessions render simultaneously.
    if (activeProgressEvents.length > 0) {
      return activeProgressEvents.map((event) => ({
        key: event.item_id,
        value: (event.percentage ?? 0) * 100,
        isIndeterminate: isConnected && (loadingModelsCount > 0 || event.percentage === null || event.percentage === 0),
      }));
    }

    // Fallback single bar: idle, or generation has started but no progress event has arrived yet (e.g.
    // while models are loading). Mirrors the previous single-bar indeterminate behavior.
    let isIndeterminate = false;
    if (isConnected && (loadingModelsCount > 0 || Boolean(queueStatus?.queue.in_progress))) {
      isIndeterminate = true;
    }
    return [{ key: 'idle', value: 0, isIndeterminate }];
  }, [activeProgressEvents, isConnected, loadingModelsCount, queueStatus?.queue.in_progress]);

  return (
    <Flex flexDir="column" gap="2px" w="full" {...containerProps}>
      {bars.map((bar) => (
        <Progress
          key={bar.key}
          value={bar.value}
          aria-label={t('accessibility.invokeProgressBar')}
          isIndeterminate={bar.isIndeterminate}
          h={2}
          w="full"
          colorScheme="invokeBlue"
          {...props}
        />
      ))}
    </Flex>
  );
};

export default memo(ProgressBar);
