import type { FlexProps, ProgressProps } from '@invoke-ai/ui-library';
import { Flex, Progress } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { $activeProgressEvents, $isConnected, $loadingModelsCount } from 'services/events/stores';

// In "fit" mode (e.g. the strip below a dockview tab label) the stack is constrained to a fixed height.
// Bars stay at FIT_BAR_HEIGHT_PX while they fit, then shrink to share the available space so they never
// overlap the label, no matter how many sessions are running.
const FIT_BAR_HEIGHT_PX = 4;
const FIT_BAR_GAP_PX = 1;

type ProgressBarProps = ProgressProps & {
  /** Applied to the Flex that stacks the per-session bars. Use for positioning (e.g. absolute). */
  containerProps?: FlexProps;
  /**
   * When set, the stacked bars are constrained to this total height (in px) and shrink to share it, so
   * they never grow past the available space (e.g. the strip below a dockview tab label).
   */
  fitHeightPx?: number;
};

type BarDescriptor = {
  key: number | string;
  value: number;
  isIndeterminate: boolean;
};

const ProgressBar = ({ containerProps, fitHeightPx, ...props }: ProgressBarProps) => {
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

  // In fit mode, cap the whole stack to the available strip and let the bars flex to share it. When the
  // bars fit at their natural height the stack is shorter than the cap; once they don't, they shrink.
  const isFit = fitHeightPx !== undefined;
  const fitContainerProps = useMemo<FlexProps | undefined>(() => {
    if (!isFit) {
      return undefined;
    }
    const naturalHeight = bars.length * FIT_BAR_HEIGHT_PX + Math.max(0, bars.length - 1) * FIT_BAR_GAP_PX;
    return { h: `${Math.min(naturalHeight, fitHeightPx)}px`, gap: `${FIT_BAR_GAP_PX}px` };
  }, [bars.length, fitHeightPx, isFit]);

  const fitBarProps: ProgressProps | undefined = isFit ? { flex: '1 1 0', minH: 0, h: 'auto' } : undefined;

  return (
    <Flex flexDir="column" gap="2px" w="full" {...fitContainerProps} {...containerProps}>
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
          {...fitBarProps}
        />
      ))}
    </Flex>
  );
};

export default memo(ProgressBar);
