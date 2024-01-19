import { Box } from '@chakra-ui/layout';
import { useDisclosure } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvBadge } from 'common/components/InvBadge/wrapper';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { InvMenuList } from 'common/components/InvMenu/InvMenuList';
import {
  InvMenu,
  InvMenuButton,
  InvMenuDivider,
} from 'common/components/InvMenu/wrapper';
import ClearQueueConfirmationAlertDialog from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { usePauseProcessor } from 'features/queue/hooks/usePauseProcessor';
import { useResumeProcessor } from 'features/queue/hooks/useResumeProcessor';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPauseFill, PiPlayFill, PiTrashSimpleBold } from 'react-icons/pi';
import { RiListCheck, RiPlayList2Fill } from 'react-icons/ri';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export const QueueActionsMenuButton = memo(() => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const clearQueueDisclosure = useDisclosure();
  const isPauseEnabled = useFeatureStatus('pauseQueue').isFeatureEnabled;
  const isResumeEnabled = useFeatureStatus('resumeQueue').isFeatureEnabled;
  const { queueSize } = useGetQueueStatusQuery(undefined, {
    selectFromResult: (res) => ({
      queueSize: res.data
        ? res.data.queue.pending + res.data.queue.in_progress
        : 0,
    }),
  });
  const { isLoading: isLoadingClearQueue, isDisabled: isDisabledClearQueue } =
    useClearQueue();
  const {
    resumeProcessor,
    isLoading: isLoadingResumeProcessor,
    isDisabled: isDisabledResumeProcessor,
  } = useResumeProcessor();
  const {
    pauseProcessor,
    isLoading: isLoadingPauseProcessor,
    isDisabled: isDisabledPauseProcessor,
  } = usePauseProcessor();
  const openQueue = useCallback(() => {
    dispatch(setActiveTab('queue'));
  }, [dispatch]);

  return (
    <Box pos="relative">
      <ClearQueueConfirmationAlertDialog disclosure={clearQueueDisclosure} />

      <InvMenu
        isOpen={isOpen}
        onOpen={onOpen}
        onClose={onClose}
        placement="bottom-end"
      >
        <InvMenuButton
          as={InvIconButton}
          aria-label="Queue Actions Menu"
          icon={<RiListCheck />}
        />
        <InvMenuList>
          <InvMenuItem
            isDestructive
            icon={<PiTrashSimpleBold size="16px" />}
            onClick={clearQueueDisclosure.onOpen}
            isLoading={isLoadingClearQueue}
            isDisabled={isDisabledClearQueue}
          >
            {t('queue.clearTooltip')}
          </InvMenuItem>
          {isResumeEnabled && (
            <InvMenuItem
              icon={<PiPlayFill size="14px" />}
              onClick={resumeProcessor}
              isLoading={isLoadingResumeProcessor}
              isDisabled={isDisabledResumeProcessor}
            >
              {t('queue.resumeTooltip')}
            </InvMenuItem>
          )}
          {isPauseEnabled && (
            <InvMenuItem
              icon={<PiPauseFill size="14px" />}
              onClick={pauseProcessor}
              isLoading={isLoadingPauseProcessor}
              isDisabled={isDisabledPauseProcessor}
            >
              {t('queue.pauseTooltip')}
            </InvMenuItem>
          )}
          <InvMenuDivider />
          <InvMenuItem icon={<RiPlayList2Fill />} onClick={openQueue}>
            {t('queue.openQueue')}
          </InvMenuItem>
        </InvMenuList>
      </InvMenu>
      {queueSize > 0 && (
        <InvBadge
          pos="absolute"
          insetInlineStart={-3}
          insetBlockStart={-1.5}
          colorScheme="invokeYellow"
          zIndex="docked"
        >
          {queueSize}
        </InvBadge>
      )}
    </Box>
  );
});

QueueActionsMenuButton.displayName = 'QueueActionsMenuButton';
