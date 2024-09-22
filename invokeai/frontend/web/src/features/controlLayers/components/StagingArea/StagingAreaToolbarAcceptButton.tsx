import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { INTERACTION_SCOPES } from 'common/hooks/interactionScopes';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  selectImageCount,
  selectSelectedImage,
  selectStagedImageIndex,
  stagingAreaImageAccepted,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';

export const StagingAreaToolbarAcceptButton = memo(() => {
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const index = useAppSelector(selectStagedImageIndex);
  const selectedImage = useAppSelector(selectSelectedImage);
  const imageCount = useAppSelector(selectImageCount);
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);
  const isCanvasActive = useStore(INTERACTION_SCOPES.canvas.$isActive);

  const { t } = useTranslation();

  const acceptSelected = useCallback(() => {
    if (!selectedImage) {
      return;
    }
    dispatch(stagingAreaImageAccepted({ index }));
  }, [dispatch, index, selectedImage]);

  useHotkeys(
    ['enter'],
    acceptSelected,
    {
      preventDefault: true,
      enabled: isCanvasActive && shouldShowStagedImage && imageCount > 1,
    },
    [isCanvasActive, shouldShowStagedImage, imageCount]
  );

  return (
    <IconButton
      tooltip={`${t('unifiedCanvas.accept')} (Enter)`}
      aria-label={`${t('unifiedCanvas.accept')} (Enter)`}
      icon={<PiCheckBold />}
      onClick={acceptSelected}
      colorScheme="invokeBlue"
      isDisabled={!selectedImage}
    />
  );
});

StagingAreaToolbarAcceptButton.displayName = 'StagingAreaToolbarAcceptButton';
