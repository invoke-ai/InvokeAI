import { IconButton, Menu, MenuButton, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SessionMenuItems } from 'common/components/SessionMenuItems';
import { useDeleteAllExceptCurrentQueueItemDialog } from 'features/queue/components/DeleteAllExceptCurrentQueueItemConfirmationAlertDialog';
import { QueueCountBadge } from 'features/queue/components/QueueCountBadge';
import { useDeleteCurrentQueueItem } from 'features/queue/hooks/useDeleteCurrentQueueItem';
import { usePauseProcessor } from 'features/queue/hooks/usePauseProcessor';
import { useResumeProcessor } from 'features/queue/hooks/useResumeProcessor';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiListBold, PiPauseFill, PiPlayFill, PiQueueBold, PiXBold, PiXCircle } from 'react-icons/pi';

export const QueueActionsMenuButton = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isPauseEnabled = useFeatureStatus('pauseQueue');
  const isResumeEnabled = useFeatureStatus('resumeQueue');
  const deleteAllExceptCurrent = useDeleteAllExceptCurrentQueueItemDialog();
  const deleteCurrentQueueItem = useDeleteCurrentQueueItem();
  const resumeProcessor = useResumeProcessor();
  const pauseProcessor = usePauseProcessor();
  const openQueue = useCallback(() => {
    dispatch(setActiveTab('queue'));
  }, [dispatch]);

  return (
    <>
      <Menu placement="bottom-end" isLazy lazyBehavior="unmount">
        <MenuButton ref={ref} as={IconButton} size="lg" aria-label="Queue Actions Menu" icon={<PiListBold />} />
        <MenuList>
          <MenuGroup title={t('common.new')}>
            <SessionMenuItems />
          </MenuGroup>
          <MenuGroup title={t('queue.queue')}>
            <MenuItem
              isDestructive
              icon={<PiXBold />}
              onClick={deleteCurrentQueueItem.trigger}
              isLoading={deleteCurrentQueueItem.isLoading}
              isDisabled={deleteCurrentQueueItem.isDisabled}
            >
              {t('queue.cancelTooltip')}
            </MenuItem>
            <MenuItem
              isDestructive
              icon={<PiXCircle />}
              onClick={deleteAllExceptCurrent.openDialog}
              isLoading={deleteAllExceptCurrent.isLoading}
              isDisabled={deleteAllExceptCurrent.isDisabled}
            >
              {t('queue.cancelAllExceptCurrentTooltip')}
            </MenuItem>
            {isResumeEnabled && (
              <MenuItem
                icon={<PiPlayFill />}
                onClick={resumeProcessor.trigger}
                isLoading={resumeProcessor.isLoading}
                isDisabled={resumeProcessor.isDisabled}
              >
                {t('queue.resumeTooltip')}
              </MenuItem>
            )}
            {isPauseEnabled && (
              <MenuItem
                icon={<PiPauseFill />}
                onClick={pauseProcessor.trigger}
                isLoading={pauseProcessor.isLoading}
                isDisabled={pauseProcessor.isDisabled}
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
