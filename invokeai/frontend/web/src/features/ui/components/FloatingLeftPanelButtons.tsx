import { ButtonGroup, Flex, Icon, IconButton, spinAnimation, Tooltip, useShiftModifier } from '@invoke-ai/ui-library';
import { ToolChooser } from 'features/controlLayers/components/Tool/ToolChooser';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useDeleteAllExceptCurrentQueueItemDialog } from 'features/queue/components/DeleteAllExceptCurrentQueueItemConfirmationAlertDialog';
import { InvokeButtonTooltip } from 'features/queue/components/InvokeButtonTooltip/InvokeButtonTooltip';
import { useDeleteCurrentQueueItem } from 'features/queue/hooks/useDeleteCurrentQueueItem';
import { useInvoke } from 'features/queue/hooks/useInvoke';
import { useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { panelRegistry } from 'features/ui/layouts/panel-registry/panelApiRegistry';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiCircleNotchBold,
  PiLightningFill,
  PiSlidersHorizontalBold,
  PiSparkleFill,
  PiXBold,
  PiXCircle,
} from 'react-icons/pi';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export const FloatingLeftPanelButtons = memo(() => {
  return (
    <Flex pos="absolute" transform="translate(0, -50%)" top="50%" insetInlineStart={2} direction="column" gap={2}>
      <ButtonGroup orientation="vertical" h={48}>
        <ToggleLeftPanelButton />
        <InvokeIconButton />
        <DeleteCurrentIconButton />
        <DeleteAllExceptCurrentIconButton />
      </ButtonGroup>
    </Flex>
  );
});

FloatingLeftPanelButtons.displayName = 'FloatingLeftPanelButtons';

export const FloatingCanvasLeftPanelButtons = memo(() => {
  return (
    <Flex pos="absolute" transform="translate(0, -50%)" top="50%" insetInlineStart={2} direction="column" gap={2}>
      <CanvasManagerProviderGate>
        <ToolChooser />
      </CanvasManagerProviderGate>
      <ButtonGroup orientation="vertical" h={48}>
        <ToggleLeftPanelButton />
        <InvokeIconButton />
        <DeleteCurrentIconButton />
        <DeleteAllExceptCurrentIconButton />
      </ButtonGroup>
    </Flex>
  );
});

FloatingCanvasLeftPanelButtons.displayName = 'FloatingCanvasLeftPanelButtons';

const ToggleLeftPanelButton = memo(() => {
  const { t } = useTranslation();
  const { tab } = useAutoLayoutContext();

  const onClick = useCallback(() => {
    if (panelRegistry.tabApi?.getTab() !== tab) {
      return;
    }
    panelRegistry.toggleLeftPanelInTab(tab);
  }, [tab]);

  return (
    <Tooltip label={t('accessibility.toggleLeftPanel')} placement="end">
      <IconButton
        aria-label={t('accessibility.toggleLeftPanel')}
        onClick={onClick}
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

const DeleteCurrentIconButton = memo(() => {
  const { t } = useTranslation();
  const deleteCurrentQueueItem = useDeleteCurrentQueueItem();

  return (
    <Tooltip label={t('queue.cancelTooltip')} placement="end">
      <IconButton
        onClick={deleteCurrentQueueItem.trigger}
        isDisabled={deleteCurrentQueueItem.isDisabled}
        isLoading={deleteCurrentQueueItem.isLoading}
        aria-label={t('queue.cancelTooltip')}
        icon={<PiXBold />}
        colorScheme="error"
        flexGrow={1}
      />
    </Tooltip>
  );
});

DeleteCurrentIconButton.displayName = 'DeleteCurrentIconButton';

const DeleteAllExceptCurrentIconButton = memo(() => {
  const { t } = useTranslation();
  const deleteAllExceptCurrent = useDeleteAllExceptCurrentQueueItemDialog();

  return (
    <Tooltip label={t('queue.cancelAllExceptCurrentTooltip')} placement="end">
      <IconButton
        isDisabled={deleteAllExceptCurrent.isDisabled}
        isLoading={deleteAllExceptCurrent.isLoading}
        aria-label={t('queue.clear')}
        icon={<PiXCircle />}
        colorScheme="error"
        onClick={deleteAllExceptCurrent.openDialog}
        flexGrow={1}
      />
    </Tooltip>
  );
});

DeleteAllExceptCurrentIconButton.displayName = 'DeleteAllExceptCurrentIconButton';
