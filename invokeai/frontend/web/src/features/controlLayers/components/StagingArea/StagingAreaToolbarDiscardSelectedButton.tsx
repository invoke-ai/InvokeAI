import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { canvasSessionReset, generateSessionReset } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useCancelQueueItem } from 'features/queue/hooks/useCancelQueueItem';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardSelectedButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const dispatch = useAppDispatch();
  const ctx = useCanvasSessionContext();
  const cancelQueueItem = useCancelQueueItem();
  const selectedItemId = useStore(ctx.$selectedItemId);

  const { t } = useTranslation();

  const discardSelected = useCallback(async () => {
    if (selectedItemId === null) {
      return;
    }
    const itemCount = ctx.$itemCount.get();
    ctx.discard(selectedItemId);
    await cancelQueueItem.trigger(selectedItemId, { withToast: false });
    if (itemCount <= 1) {
      if (ctx.session.type === 'advanced') {
        dispatch(canvasSessionReset());
      } else {
        // ctx.session.type === 'simple'
        dispatch(generateSessionReset());
      }
    }
  }, [selectedItemId, ctx, cancelQueueItem, dispatch]);

  return (
    <IconButton
      tooltip={t('controlLayers.stagingArea.discard')}
      aria-label={t('controlLayers.stagingArea.discard')}
      icon={<PiXBold />}
      onClick={discardSelected}
      colorScheme="invokeBlue"
      isDisabled={selectedItemId === null || cancelQueueItem.isDisabled || isDisabled}
      isLoading={cancelQueueItem.isLoading}
    />
  );
});

StagingAreaToolbarDiscardSelectedButton.displayName = 'StagingAreaToolbarDiscardSelectedButton';
