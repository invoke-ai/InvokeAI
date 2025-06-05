import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { canvasSessionGenerationFinished } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useDeleteQueueItemMutation } from 'services/api/endpoints/queue';

export const StagingAreaToolbarDiscardSelectedButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const dispatch = useAppDispatch();
  const ctx = useCanvasSessionContext();
  const [deleteQueueItem] = useDeleteQueueItemMutation();
  const selectedItemId = useStore(ctx.$selectedItemId);

  const { t } = useTranslation();

  const discardSelected = useCallback(() => {
    if (selectedItemId === null) {
      return;
    }
    const itemCount = ctx.$itemCount.get();
    deleteQueueItem({ item_id: selectedItemId });
    if (itemCount <= 1) {
      dispatch(canvasSessionGenerationFinished());
    }
  }, [selectedItemId, ctx.$itemCount, deleteQueueItem, dispatch]);

  return (
    <IconButton
      tooltip={t('controlLayers.stagingArea.discard')}
      aria-label={t('controlLayers.stagingArea.discard')}
      icon={<PiXBold />}
      onClick={discardSelected}
      colorScheme="invokeBlue"
      fontSize={16}
      isDisabled={selectedItemId === null || isDisabled}
    />
  );
});

StagingAreaToolbarDiscardSelectedButton.displayName = 'StagingAreaToolbarDiscardSelectedButton';
