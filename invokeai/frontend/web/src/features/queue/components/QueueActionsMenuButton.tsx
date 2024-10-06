import { IconButton, Menu, MenuButton, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import {
  useNewCanvasSession,
  useNewGallerySession,
} from 'features/controlLayers/components/NewSessionConfirmationAlertDialog';
import { useClearQueue } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { QueueCountBadge } from 'features/queue/components/QueueCountBadge';
import { usePauseProcessor } from 'features/queue/hooks/usePauseProcessor';
import { useResumeProcessor } from 'features/queue/hooks/useResumeProcessor';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiImageBold,
  PiListBold,
  PiPaintBrushBold,
  PiPauseFill,
  PiPlayFill,
  PiQueueBold,
  PiTrashSimpleBold,
  PiXBold,
} from 'react-icons/pi';

export const QueueActionsMenuButton = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isPauseEnabled = useFeatureStatus('pauseQueue');
  const isResumeEnabled = useFeatureStatus('resumeQueue');
  const { newGallerySessionWithDialog } = useNewGallerySession();
  const { newCanvasSessionWithDialog } = useNewCanvasSession();
  const clearQueue = useClearQueue();
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
            <MenuItem icon={<PiImageBold />} onClick={newGallerySessionWithDialog}>
              {t('controlLayers.newGallerySession')}
            </MenuItem>
            <MenuItem icon={<PiPaintBrushBold />} onClick={newCanvasSessionWithDialog}>
              {t('controlLayers.newCanvasSession')}
            </MenuItem>
          </MenuGroup>
          <MenuGroup title={t('queue.queue')}>
            <MenuItem
              isDestructive
              icon={<PiXBold />}
              onClick={clearQueue.openDialog}
              isLoading={clearQueue.isLoading}
              isDisabled={clearQueue.isDisabled}
            >
              {t('queue.cancelTooltip')}
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
