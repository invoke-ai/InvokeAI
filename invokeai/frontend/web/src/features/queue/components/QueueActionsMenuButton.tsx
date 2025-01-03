import { IconButton, Menu, MenuButton, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SessionMenuItems } from 'common/components/SessionMenuItems';
import { useCancelAllExceptCurrentQueueItemDialog } from 'features/queue/components/CancelAllExceptCurrentQueueItemConfirmationAlertDialog';
import { useClearQueueDialog } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { QueueCountBadge } from 'features/queue/components/QueueCountBadge';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { usePauseProcessor } from 'features/queue/hooks/usePauseProcessor';
import { useResumeProcessor } from 'features/queue/hooks/useResumeProcessor';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiListBold,
  PiPauseFill,
  PiPlayFill,
  PiQueueBold,
  PiTrashSimpleBold,
  PiXBold,
  PiXCircle,
} from 'react-icons/pi';

export const QueueActionsMenuButton = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isPauseEnabled = useFeatureStatus('pauseQueue');
  const isResumeEnabled = useFeatureStatus('resumeQueue');
  const cancelAllExceptCurrent = useCancelAllExceptCurrentQueueItemDialog();
  const cancelCurrent = useCancelCurrentQueueItem();
  const clearQueue = useClearQueueDialog();
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
    <>
      <Menu placement="bottom-end">
        <MenuButton ref={ref} as={IconButton} size="lg" aria-label="Queue Actions Menu" icon={<PiListBold />} />
        <MenuList>
          <MenuGroup title={t('common.new')}>
            <SessionMenuItems />
          </MenuGroup>
          <MenuGroup title={t('queue.queue')}>
            <MenuItem
              isDestructive
              icon={<PiXBold />}
              onClick={cancelCurrent.cancelQueueItem}
              isLoading={cancelCurrent.isLoading}
              isDisabled={cancelCurrent.isDisabled}
            >
              {t('queue.cancelTooltip')}
            </MenuItem>
            <MenuItem
              isDestructive
              icon={<PiXCircle />}
              onClick={cancelAllExceptCurrent.openDialog}
              isLoading={cancelAllExceptCurrent.isLoading}
              isDisabled={cancelAllExceptCurrent.isDisabled}
            >
              {t('queue.cancelAllExceptCurrentTooltip')}
            </MenuItem>
            <MenuItem
              isDestructive
              icon={<PiTrashSimpleBold />}
              onClick={clearQueue.openDialog}
              isLoading={clearQueue.isLoading}
              isDisabled={clearQueue.isDisabled}
            >
              {t('queue.clearTooltip')}
            </MenuItem>
            {isResumeEnabled && (
              <MenuItem
                icon={<PiPlayFill />}
                onClick={resumeProcessor}
                isLoading={isLoadingResumeProcessor}
                isDisabled={isDisabledResumeProcessor}
              >
                {t('queue.resumeTooltip')}
              </MenuItem>
            )}
            {isPauseEnabled && (
              <MenuItem
                icon={<PiPauseFill />}
                onClick={pauseProcessor}
                isLoading={isLoadingPauseProcessor}
                isDisabled={isDisabledPauseProcessor}
              >
                {t('queue.pauseTooltip')}
              </MenuItem>
            )}
            <MenuItem icon={<PiQueueBold />} onClick={openQueue}>
              {t('queue.openQueue')}
            </MenuItem>
          </MenuGroup>
        </MenuList>
      </Menu>
      {/* The badge is dynamically positioned, needs a ref to the target element */}
      <QueueCountBadge targetRef={ref} />
    </>
  );
});

QueueActionsMenuButton.displayName = 'QueueActionsMenuButton';
