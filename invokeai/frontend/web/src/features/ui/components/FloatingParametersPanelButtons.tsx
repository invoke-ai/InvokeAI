import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { ButtonGroup, Flex, Icon, IconButton, Portal, spinAnimation } from '@invoke-ai/ui-library';
import CancelCurrentQueueItemIconButton from 'features/queue/components/CancelCurrentQueueItemIconButton';
import { QueueButtonTooltip } from 'features/queue/components/QueueButtonTooltip';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { useQueueBack } from 'features/queue/hooks/useQueueBack';
import type { UsePanelReturn } from 'features/ui/hooks/usePanel';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCircleNotchBold, PiSlidersHorizontalBold, PiTrashSimpleBold } from 'react-icons/pi';
import { RiSparklingFill } from 'react-icons/ri';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const floatingButtonStyles: SystemStyleObject = {
  borderStartRadius: 0,
  flexGrow: 1,
};

type Props = {
  panelApi: UsePanelReturn;
};

const FloatingSidePanelButtons = (props: Props) => {
  const { t } = useTranslation();
  const { queueBack, isLoading, isDisabled } = useQueueBack();
  const clearQueue = useClearQueue();
  const { data: queueStatus } = useGetQueueStatusQuery();

  const queueButtonIcon = useMemo(() => {
    const isProcessing = (queueStatus?.queue.in_progress ?? 0) > 0;
    if (!isDisabled && isProcessing) {
      return <Icon boxSize={6} as={PiCircleNotchBold} animation={spinAnimation} />;
    }
    return <RiSparklingFill size="16px" />;
  }, [isDisabled, queueStatus]);

  if (!props.panelApi.isCollapsed) {
    return null;
  }

  return (
    <Portal>
      <Flex
        pos="absolute"
        transform="translate(0, -50%)"
        minW={8}
        top="50%"
        insetInlineStart="63px"
        direction="column"
        gap={2}
        h={48}
        zIndex={11}
      >
        <ButtonGroup orientation="vertical" flexGrow={3}>
          <IconButton
            tooltip={t('accessibility.showOptionsPanel')}
            aria-label={t('accessibility.showOptionsPanel')}
            onClick={props.panelApi.expand}
            sx={floatingButtonStyles}
            icon={<PiSlidersHorizontalBold size="16px" />}
          />
          <QueueButtonTooltip>
            <IconButton
              aria-label={t('queue.queueBack')}
              onClick={queueBack}
              isLoading={isLoading}
              isDisabled={isDisabled}
              icon={queueButtonIcon}
              colorScheme="invokeYellow"
              sx={floatingButtonStyles}
            />
          </QueueButtonTooltip>
          <CancelCurrentQueueItemIconButton sx={floatingButtonStyles} />
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
          sx={floatingButtonStyles}
        />
      </Flex>
    </Portal>
  );
};

export default memo(FloatingSidePanelButtons);
