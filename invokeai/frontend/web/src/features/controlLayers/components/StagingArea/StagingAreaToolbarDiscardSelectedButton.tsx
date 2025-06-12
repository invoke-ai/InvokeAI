import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { canvasSessionReset, generateSessionReset } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useDeleteQueueItem } from 'features/queue/hooks/useDeleteQueueItem';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardSelectedButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const dispatch = useAppDispatch();
  const ctx = useCanvasSessionContext();
  const deleteQueueItem = useDeleteQueueItem();
  const selectedItemId = useStore(ctx.$selectedItemId);

  const { t } = useTranslation();

  const discardSelected = useCallback(async () => {
    if (selectedItemId === null) {
      return;
    }
    await deleteQueueItem.trigger(selectedItemId);
    const itemCount = ctx.$itemCount.get();
    if (itemCount <= 1) {
      if (ctx.session.type === 'advanced') {
        dispatch(canvasSessionReset());
      } else {
        // ctx.session.type === 'simple'
        dispatch(generateSessionReset());
      }
    }
  }, [selectedItemId, deleteQueueItem, ctx.$itemCount, ctx.session.type, dispatch]);

  return (
    <IconButton
      tooltip={t('controlLayers.stagingArea.discard')}
      aria-label={t('controlLayers.stagingArea.discard')}
      icon={<PiXBold />}
      onClick={discardSelected}
      colorScheme="invokeBlue"
      fontSize={16}
      isDisabled={selectedItemId === null || deleteQueueItem.isDisabled || isDisabled}
      isLoading={deleteQueueItem.isLoading}
    />
  );
});

StagingAreaToolbarDiscardSelectedButton.displayName = 'StagingAreaToolbarDiscardSelectedButton';
