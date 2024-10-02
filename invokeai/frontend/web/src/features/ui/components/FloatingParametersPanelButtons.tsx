import { ButtonGroup, Flex, Icon, IconButton, spinAnimation, Tooltip, useShiftModifier } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ToolChooser } from 'features/controlLayers/components/Tool/ToolChooser';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useClearQueue } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { QueueButtonTooltip } from 'features/queue/components/QueueButtonTooltip';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { useInvoke } from 'features/queue/hooks/useInvoke';
import type { UsePanelReturn } from 'features/ui/hooks/usePanel';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiCircleNotchBold,
  PiLightningFill,
  PiSlidersHorizontalBold,
  PiSparkleFill,
  PiTrashSimpleBold,
  PiXBold,
} from 'react-icons/pi';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

type Props = {
  panelApi: UsePanelReturn;
};

const FloatingSidePanelButtons = (props: Props) => {
  const { t } = useTranslation();
  const queue = useInvoke();
  const shift = useShiftModifier();
  const tab = useAppSelector(selectActiveTab);
  const imageViewer = useImageViewer();
  const clearQueue = useClearQueue();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const cancelCurrent = useCancelCurrentQueueItem();

  const queueButtonIcon = useMemo(() => {
    const isProcessing = (queueStatus?.queue.in_progress ?? 0) > 0;
    if (!queue.isDisabled && isProcessing) {
      return <Icon boxSize={6} as={PiCircleNotchBold} animation={spinAnimation} />;
    }
    if (shift) {
      return <PiLightningFill />;
    }
    return <PiSparkleFill />;
  }, [queue.isDisabled, queueStatus?.queue.in_progress, shift]);

  return (
    <Flex pos="absolute" transform="translate(0, -50%)" top="50%" insetInlineStart={2} direction="column" gap={2}>
      {tab === 'canvas' && !imageViewer.isOpen && (
        <CanvasManagerProviderGate>
          <ToolChooser />
        </CanvasManagerProviderGate>
      )}
      <ButtonGroup orientation="vertical" h={48}>
        <Tooltip label={t('accessibility.toggleLeftPanel')} placement="end">
          <IconButton
            aria-label={t('accessibility.toggleLeftPanel')}
            onClick={props.panelApi.toggle}
            icon={<PiSlidersHorizontalBold />}
            flexGrow={1}
          />
        </Tooltip>
        <QueueButtonTooltip prepend={shift} placement="end">
          <IconButton
            aria-label={t('queue.queueBack')}
            onClick={shift ? queue.queueFront : queue.queueBack}
            isLoading={queue.isLoading}
            isDisabled={queue.isDisabled}
            icon={queueButtonIcon}
            colorScheme="invokeYellow"
            flexGrow={1}
          />
        </QueueButtonTooltip>
        <Tooltip label={t('queue.cancelTooltip')} placement="end">
          <IconButton
            isDisabled={cancelCurrent.isDisabled}
            isLoading={cancelCurrent.isLoading}
            aria-label={t('queue.cancelTooltip')}
            icon={<PiXBold />}
            onClick={cancelCurrent.cancelQueueItem}
            colorScheme="error"
            flexGrow={1}
          />
        </Tooltip>

        <Tooltip label={t('queue.clearTooltip')} placement="end">
          <IconButton
            isDisabled={clearQueue.isDisabled}
            isLoading={clearQueue.isLoading}
            aria-label={t('queue.clearTooltip')}
            icon={<PiTrashSimpleBold />}
            colorScheme="error"
            onClick={clearQueue.openDialog}
            data-testid={t('queue.clear')}
            flexGrow={1}
          />
        </Tooltip>
      </ButtonGroup>
    </Flex>
  );
};

export default memo(FloatingSidePanelButtons);
