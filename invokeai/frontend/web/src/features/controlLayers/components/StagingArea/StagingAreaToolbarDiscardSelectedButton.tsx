import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectImageCount,
  selectSelectedImage,
  selectStagedImageIndex,
  stagingAreaReset,
  stagingAreaStagedImageDiscarded,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardSelectedButton = memo(() => {
  const dispatch = useAppDispatch();
  const index = useAppSelector(selectStagedImageIndex);
  const selectedImage = useAppSelector(selectSelectedImage);
  const imageCount = useAppSelector(selectImageCount);

  const { t } = useTranslation();

  const discardSelected = useCallback(() => {
    if (!selectedImage) {
      return;
    }
    if (imageCount === 1) {
      dispatch(stagingAreaReset());
    } else {
      dispatch(stagingAreaStagedImageDiscarded({ index }));
    }
  }, [selectedImage, imageCount, dispatch, index]);

  return (
    <IconButton
      tooltip={t('controlLayers.stagingArea.discard')}
      aria-label={t('controlLayers.stagingArea.discard')}
      icon={<PiXBold />}
      onClick={discardSelected}
      colorScheme="invokeBlue"
      fontSize={16}
      isDisabled={!selectedImage}
    />
  );
});

StagingAreaToolbarDiscardSelectedButton.displayName = 'StagingAreaToolbarDiscardSelectedButton';
