import {
  Badge,
  Box,
  IconButton,
  Menu,
  MenuButton,
  MenuDivider,
  MenuItem,
  MenuList,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
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
  const isPauseEnabled = useFeatureStatus('pauseQueue');
  const isResumeEnabled = useFeatureStatus('resumeQueue');
  const { queueSize } = useGetQueueStatusQuery(undefined, {
    selectFromResult: (res) => ({
      queueSize: res.data ? res.data.queue.pending + res.data.queue.in_progress : 0,
    }),
  });
  const { isLoading: isLoadingClearQueue, isDisabled: isDisabledClearQueue } = useClearQueue();
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

      <Menu isOpen={isOpen} onOpen={onOpen} onClose={onClose} placement="bottom-end">
        <MenuButton as={IconButton} aria-label="Queue Actions Menu" icon={<RiListCheck />} />
        <MenuList>
          <MenuItem
            isDestructive
            icon={<PiTrashSimpleBold size="16px" />}
            onClick={clearQueueDisclosure.onOpen}
            isLoading={isLoadingClearQueue}
            isDisabled={isDisabledClearQueue}
          >
            {t('queue.clearTooltip')}
          </MenuItem>
          {isResumeEnabled && (
            <MenuItem
              icon={<PiPlayFill size="14px" />}
              onClick={resumeProcessor}
              isLoading={isLoadingResumeProcessor}
              isDisabled={isDisabledResumeProcessor}
            >
              {t('queue.resumeTooltip')}
            </MenuItem>
          )}
          {isPauseEnabled && (
            <MenuItem
              icon={<PiPauseFill size="14px" />}
              onClick={pauseProcessor}
              isLoading={isLoadingPauseProcessor}
              isDisabled={isDisabledPauseProcessor}
            >
              {t('queue.pauseTooltip')}
            </MenuItem>
          )}
          <MenuDivider />
          <MenuItem icon={<RiPlayList2Fill />} onClick={openQueue}>
            {t('queue.openQueue')}
          </MenuItem>
        </MenuList>
      </Menu>
      {queueSize > 0 && (
        <Badge pos="absolute" insetInlineStart={-3} insetBlockStart={-1.5} colorScheme="invokeYellow" zIndex="docked">
          {queueSize}
        </Badge>
      )}
    </Box>
  );
});

QueueActionsMenuButton.displayName = 'QueueActionsMenuButton';
