import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { canvasSessionReset, generateSessionReset } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useDeleteQueueItemsByDestination } from 'features/queue/hooks/useDeleteQueueItemsByDestination';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardAllButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const ctx = useCanvasSessionContext();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const deleteQueueItemsByDestination = useDeleteQueueItemsByDestination();

  const discardAll = useCallback(() => {
    deleteQueueItemsByDestination.trigger(ctx.session.id);
    if (ctx.session.type === 'advanced') {
      dispatch(canvasSessionReset());
    } else {
      // ctx.session.type === 'simple'
      dispatch(generateSessionReset());
    }
  }, [deleteQueueItemsByDestination, ctx.session.id, ctx.session.type, dispatch]);

  return (
    <IconButton
      tooltip={`${t('controlLayers.stagingArea.discardAll')} (Esc)`}
      aria-label={t('controlLayers.stagingArea.discardAll')}
      icon={<PiTrashSimpleBold />}
      onClick={discardAll}
      colorScheme="error"
      fontSize={16}
      isDisabled={isDisabled || deleteQueueItemsByDestination.isDisabled}
      isLoading={deleteQueueItemsByDestination.isLoading}
    />
  );
});

StagingAreaToolbarDiscardAllButton.displayName = 'StagingAreaToolbarDiscardAllButton';
