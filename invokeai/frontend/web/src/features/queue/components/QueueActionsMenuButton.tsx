import {
  Badge,
  Box,
  IconButton,
  Menu,
  MenuButton,
  MenuDivider,
  MenuItem,
  MenuList,
  Portal,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type { Coordinate } from 'features/controlLayers/store/types';
import { useClearQueueConfirmationAlertDialog } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { usePauseProcessor } from 'features/queue/hooks/usePauseProcessor';
import { useResumeProcessor } from 'features/queue/hooks/useResumeProcessor';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPauseFill, PiPlayFill, PiTrashSimpleBold } from 'react-icons/pi';
import { RiListCheck, RiPlayList2Fill } from 'react-icons/ri';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export const QueueActionsMenuButton = memo(() => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [badgePos, setBadgePos] = useState<Coordinate | null>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const dialogState = useClearQueueConfirmationAlertDialog();
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

  useEffect(() => {
    if (menuButtonRef.current) {
      const { x, y } = menuButtonRef.current.getBoundingClientRect();
      setBadgePos({ x: x - 10, y: y - 10 });
    }
  }, []);

  return (
    <Box pos="relative">
      <Menu isOpen={isOpen} onOpen={onOpen} onClose={onClose} placement="bottom-end">
        <MenuButton ref={menuButtonRef} as={IconButton} aria-label="Queue Actions Menu" icon={<RiListCheck />} />
        <MenuList>
          <MenuItem
            isDestructive
            icon={<PiTrashSimpleBold size="16px" />}
            onClick={dialogState.setTrue}
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
      {queueSize > 0 && badgePos !== null && (
        <Portal>
          <Badge
            pos="absolute"
            insetInlineStart={badgePos.x}
            insetBlockStart={badgePos.y}
            colorScheme="invokeYellow"
            zIndex="docked"
          >
            {queueSize}
          </Badge>
        </Portal>
      )}
    </Box>
  );
});

QueueActionsMenuButton.displayName = 'QueueActionsMenuButton';
