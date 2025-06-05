import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import {
  selectImageCount,
  selectSelectedImage,
  selectStagedImageIndex,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useDeleteQueueItemMutation } from 'services/api/endpoints/queue';

export const StagingAreaToolbarDiscardSelectedButton = memo(() => {
  const ctx = useCanvasSessionContext();
  const dispatch = useAppDispatch();
  const [deleteQueueItem] = useDeleteQueueItemMutation();
  const selectedItemId = useStore(ctx.$selectedItemId);
  const index = useAppSelector(selectStagedImageIndex);
  const selectedImage = useAppSelector(selectSelectedImage);
  const imageCount = useAppSelector(selectImageCount);

  const { t } = useTranslation();

  const discardSelected = useCallback(() => {
    if (selectedItemId === null) {
      return;
    }
    deleteQueueItem({ item_id: selectedItemId });
    // if (imageCount === 1) {
    //   dispatch(stagingAreaReset());
    // } else {
    //   dispatch(stagingAreaStagedImageDiscarded({ index }));
    // }
  }, [selectedItemId, deleteQueueItem]);

  return (
    <IconButton
      tooltip={t('controlLayers.stagingArea.discard')}
      aria-label={t('controlLayers.stagingArea.discard')}
      icon={<PiXBold />}
      onClick={discardSelected}
      colorScheme="invokeBlue"
      fontSize={16}
      isDisabled={selectedItemId === null}
    />
  );
});

StagingAreaToolbarDiscardSelectedButton.displayName = 'StagingAreaToolbarDiscardSelectedButton';
