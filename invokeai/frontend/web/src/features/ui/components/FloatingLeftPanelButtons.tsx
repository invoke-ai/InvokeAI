import { ButtonGroup, Flex, Icon, IconButton, spinAnimation, Tooltip, useShiftModifier } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ToolChooser } from 'features/controlLayers/components/Tool/ToolChooser';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectCanvasSession } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useCancelAllExceptCurrentQueueItemDialog } from 'features/queue/components/CancelAllExceptCurrentQueueItemConfirmationAlertDialog';
import { useClearQueueDialog } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { InvokeButtonTooltip } from 'features/queue/components/InvokeButtonTooltip/InvokeButtonTooltip';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { useInvoke } from 'features/queue/hooks/useInvoke';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiCircleNotchBold,
  PiLightningFill,
  PiSlidersHorizontalBold,
  PiSparkleFill,
  PiTrashSimpleBold,
  PiXBold,
  PiXCircle,
} from 'react-icons/pi';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export const FloatingLeftPanelButtons = memo((props: { onToggle: () => void }) => {
  const tab = useAppSelector(selectActiveTab);
  const session = useAppSelector(selectCanvasSession);

  return (
    <Flex pos="absolute" transform="translate(0, -50%)" top="50%" insetInlineStart={2} direction="column" gap={2}>
      {tab === 'canvas' && session?.type === 'advanced' && (
        <CanvasManagerProviderGate>
          <ToolChooser />
        </CanvasManagerProviderGate>
      )}
      <ButtonGroup orientation="vertical" h={48}>
        <ToggleLeftPanelButton onToggle={props.onToggle} />
        <InvokeIconButton />
        <CancelCurrentIconButton />
        <CancelAllExceptCurrentIconButton />
      </ButtonGroup>
    </Flex>
  );
});

FloatingLeftPanelButtons.displayName = 'FloatingLeftPanelButtons';

const ToggleLeftPanelButton = memo((props: { onToggle: () => void }) => {
  const { t } = useTranslation();
  return (
    <Tooltip label={t('accessibility.toggleLeftPanel')} placement="end">
      <IconButton
        aria-label={t('accessibility.toggleLeftPanel')}
        onClick={props.onToggle}
        icon={<PiSlidersHorizontalBold />}
        flexGrow={1}
      />
    </Tooltip>
  );
});
ToggleLeftPanelButton.displayName = 'ToggleLeftPanelButton';

const InvokeIconButton = memo(() => {
  const { t } = useTranslation();
  const queue = useInvoke();
  const shift = useShiftModifier();

  return (
    <InvokeButtonTooltip prepend={shift} placement="end">
      <IconButton
        aria-label={t('queue.queueBack')}
        onClick={shift ? queue.enqueueFront : queue.enqueueBack}
        isLoading={queue.isLoading}
        isDisabled={queue.isDisabled}
        icon={<InvokeIconButtonIcon />}
        colorScheme="invokeYellow"
        flexGrow={1}
      />
    </InvokeButtonTooltip>
  );
});
InvokeIconButton.displayName = 'InvokeIconButton';

const InvokeIconButtonIcon = memo(() => {
  const shift = useShiftModifier();
  const queue = useInvoke();
  const { isProcessing } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { isProcessing: false };
      }
      return { isProcessing: data.queue.in_progress > 0 };
    },
  });

  if (!queue.isDisabled && isProcessing) {
    return <Icon boxSize={6} as={PiCircleNotchBold} animation={spinAnimation} />;
  }

  if (shift) {
    return <PiLightningFill />;
  }

  return <PiSparkleFill />;
});
InvokeIconButtonIcon.displayName = 'InvokeIconButtonIcon';

const CancelCurrentIconButton = memo(() => {
  const { t } = useTranslation();
  const cancelCurrentQueueItem = useCancelCurrentQueueItem();

  return (
    <Tooltip label={t('queue.cancelTooltip')} placement="end">
      <IconButton
        isDisabled={cancelCurrentQueueItem.isDisabled}
        isLoading={cancelCurrentQueueItem.isLoading}
        aria-label={t('queue.cancelTooltip')}
        icon={<PiXBold />}
        onClick={cancelCurrentQueueItem.cancelQueueItem}
        colorScheme="error"
        flexGrow={1}
      />
    </Tooltip>
  );
});

CancelCurrentIconButton.displayName = 'CancelCurrentIconButton';

const CancelAndClearAllIconButton = memo(() => {
  const { t } = useTranslation();
  const clearQueue = useClearQueueDialog();

  return (
    <Tooltip label={t('queue.clearTooltip')} placement="end">
      <IconButton
        isDisabled={clearQueue.isDisabled}
        isLoading={clearQueue.isLoading}
        aria-label={t('queue.clearTooltip')}
        icon={<PiTrashSimpleBold />}
        colorScheme="error"
        onClick={clearQueue.openDialog}
        flexGrow={1}
      />
    </Tooltip>
  );
});

CancelAndClearAllIconButton.displayName = 'CancelAndClearAllIconButton';

const CancelAllExceptCurrentIconButton = memo(() => {
  const { t } = useTranslation();
  const cancelAllExceptCurrent = useCancelAllExceptCurrentQueueItemDialog();

  return (
    <Tooltip label={t('queue.cancelAllExceptCurrentTooltip')} placement="end">
      <IconButton
        isDisabled={cancelAllExceptCurrent.isDisabled}
        isLoading={cancelAllExceptCurrent.isLoading}
        aria-label={t('queue.clear')}
        icon={<PiXCircle />}
        colorScheme="error"
        onClick={cancelAllExceptCurrent.openDialog}
        flexGrow={1}
      />
    </Tooltip>
  );
});

CancelAllExceptCurrentIconButton.displayName = 'CancelAllExceptCurrentIconButton';
