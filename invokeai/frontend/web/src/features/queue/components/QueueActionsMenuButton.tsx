import {
  Badge,
  IconButton,
  Menu,
  MenuButton,
  MenuDivider,
  MenuItem,
  MenuList,
  Portal,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch } from 'app/store/storeHooks';
import type { Coordinate } from 'features/controlLayers/store/types';
import { useClearQueueConfirmationAlertDialog } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { usePauseProcessor } from 'features/queue/hooks/usePauseProcessor';
import { useResumeProcessor } from 'features/queue/hooks/useResumeProcessor';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { $isParametersPanelOpen, setActiveTab } from 'features/ui/store/uiSlice';
import type { RefObject } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPauseFill, PiPlayFill, PiTrashSimpleBold } from 'react-icons/pi';
import { RiListCheck, RiPlayList2Fill } from 'react-icons/ri';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

type Props = {
  containerRef: RefObject<HTMLDivElement>;
};

export const QueueActionsMenuButton = memo(({ containerRef }: Props) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [badgePos, setBadgePos] = useState<Coordinate | null>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const isParametersPanelOpen = useStore($isParametersPanelOpen);
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
    if (!containerRef.current || !menuButtonRef.current) {
      return;
    }

    const container = containerRef.current;
    const menuButton = menuButtonRef.current;

    const cb = () => {
      if (!$isParametersPanelOpen.get()) {
        return;
      }
      const { x, y } = menuButton.getBoundingClientRect();
      setBadgePos({ x: x - 10, y: y - 10 });
    };

    // // update badge position on resize
    const resizeObserver = new ResizeObserver(cb);
    resizeObserver.observe(container);
    cb();

    return () => {
      resizeObserver.disconnect();
    };
  }, [containerRef]);

  return (
    <>
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
      {queueSize > 0 && badgePos !== null && isParametersPanelOpen && (
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
    </>
  );
});

QueueActionsMenuButton.displayName = 'QueueActionsMenuButton';
