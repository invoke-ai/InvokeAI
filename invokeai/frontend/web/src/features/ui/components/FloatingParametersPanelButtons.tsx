import { ButtonGroup, Flex, Icon, IconButton, spinAnimation, useShiftModifier } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ToolChooser } from 'features/controlLayers/components/Tool/ToolChooser';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import CancelCurrentQueueItemIconButton from 'features/queue/components/CancelCurrentQueueItemIconButton';
import { useClearQueue } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { QueueButtonTooltip } from 'features/queue/components/QueueButtonTooltip';
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
    <Flex
      pos="absolute"
      transform="translate(0, -50%)"
      top="50%"
      insetInlineStart={2}
      direction="column"
      gap={2}
      zIndex={11}
    >
      {tab === 'canvas' && !imageViewer.isOpen && (
        <CanvasManagerProviderGate>
          <ToolChooser />
        </CanvasManagerProviderGate>
      )}
      <ButtonGroup orientation="vertical">
        <IconButton
          tooltip={t('accessibility.showOptionsPanel')}
          aria-label={t('accessibility.showOptionsPanel')}
          onClick={props.panelApi.toggle}
          icon={<PiSlidersHorizontalBold />}
        />
        <QueueButtonTooltip prepend={shift}>
          <IconButton
            aria-label={t('queue.queueBack')}
            onClick={shift ? queue.queueFront : queue.queueBack}
            isLoading={queue.isLoading}
            isDisabled={queue.isDisabled}
            icon={queueButtonIcon}
            colorScheme="invokeYellow"
          />
        </QueueButtonTooltip>
        <CancelCurrentQueueItemIconButton />
      </ButtonGroup>
      <IconButton
        isDisabled={clearQueue.isDisabled}
        isLoading={clearQueue.isLoading}
        aria-label={t('queue.clear')}
        tooltip={t('queue.clearTooltip')}
        icon={<PiTrashSimpleBold />}
        colorScheme="error"
        onClick={clearQueue.openDialog}
        data-testid={t('queue.clear')}
      />
    </Flex>
  );
};

export default memo(FloatingSidePanelButtons);
