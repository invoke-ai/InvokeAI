import { IconButton, Menu, MenuButton, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { SessionMenuItems } from 'common/components/SessionMenuItems';
import { selectActiveTab } from 'features/controlLayers/store/selectors';
import { useCancelAllExceptCurrentQueueItemDialog } from 'features/queue/components/CancelAllExceptCurrentQueueItemConfirmationAlertDialog';
import { QueueCountBadge } from 'features/queue/components/QueueCountBadge';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { usePauseProcessor } from 'features/queue/hooks/usePauseProcessor';
import { useResumeProcessor } from 'features/queue/hooks/useResumeProcessor';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiListBold, PiPauseFill, PiPlayFill, PiQueueBold, PiTrashBold, PiXBold, PiXCircle } from 'react-icons/pi';

import { useClearQueueDialog } from './ClearQueueConfirmationAlertDialog';

export const QueueActionsMenuButton = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const { t } = useTranslation();
  const cancelAllExceptCurrent = useCancelAllExceptCurrentQueueItemDialog();
  const cancelCurrentQueueItem = useCancelCurrentQueueItem();
  const clearQueue = useClearQueueDialog();
  const resumeProcessor = useResumeProcessor();
  const pauseProcessor = usePauseProcessor();
  const openQueue = useCallback(() => {
    navigationApi.switchToTab('queue');
  }, []);
  const tab = useAppSelector(selectActiveTab);

  const cancelCurrentQueueItemWithToast = useCallback(() => {
    cancelCurrentQueueItem.trigger({ withToast: true });
  }, [cancelCurrentQueueItem]);

  return (
    <>
      <Menu placement="bottom-end" isLazy lazyBehavior="unmount">
        <MenuButton ref={ref} as={IconButton} size="lg" aria-label="Queue Actions Menu" icon={<PiListBold />} />
        <MenuList>
          {(tab === 'canvas' || tab === 'generate') && (
            <MenuGroup title={t('common.new')}>
              <SessionMenuItems />
            </MenuGroup>
          )}
          <MenuGroup title={t('queue.queue')}>
            <MenuItem
              isDestructive
              icon={<PiXBold />}
              onClick={cancelCurrentQueueItemWithToast}
              isLoading={cancelCurrentQueueItem.isLoading}
              isDisabled={cancelCurrentQueueItem.isDisabled}
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
              icon={<PiTrashBold />}
              onClick={clearQueue.openDialog}
              isLoading={clearQueue.isLoading}
              isDisabled={clearQueue.isDisabled}
            >
              {t('queue.clearTooltip')}
            </MenuItem>
            <MenuItem
              icon={<PiPlayFill />}
              onClick={resumeProcessor.trigger}
              isLoading={resumeProcessor.isLoading}
              isDisabled={resumeProcessor.isDisabled}
            >
              {t('queue.resumeTooltip')}
            </MenuItem>
            <MenuItem
              icon={<PiPauseFill />}
              onClick={pauseProcessor.trigger}
              isLoading={pauseProcessor.isLoading}
              isDisabled={pauseProcessor.isDisabled}
            >
              {t('queue.pauseTooltip')}
            </MenuItem>
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
