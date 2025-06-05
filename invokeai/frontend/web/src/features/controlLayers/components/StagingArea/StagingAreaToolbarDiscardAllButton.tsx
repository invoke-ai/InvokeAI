import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { canvasSessionGenerationFinished } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useDeleteQueueItemsByDestinationMutation } from 'services/api/endpoints/queue';

export const StagingAreaToolbarDiscardAllButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const ctx = useCanvasSessionContext();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [deleteByDestination] = useDeleteQueueItemsByDestinationMutation();

  const discardAll = useCallback(() => {
    deleteByDestination({ destination: ctx.session.id });
    dispatch(canvasSessionGenerationFinished());
  }, [deleteByDestination, ctx.session.id, dispatch]);

  return (
    <IconButton
      tooltip={`${t('controlLayers.stagingArea.discardAll')} (Esc)`}
      aria-label={t('controlLayers.stagingArea.discardAll')}
      icon={<PiTrashSimpleBold />}
      onClick={discardAll}
      colorScheme="error"
      fontSize={16}
      isDisabled={isDisabled}
    />
  );
});

StagingAreaToolbarDiscardAllButton.displayName = 'StagingAreaToolbarDiscardAllButton';
