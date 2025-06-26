import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';
import { copyImage } from 'services/api/endpoints/images';

const TOAST_ID = 'SAVE_STAGING_AREA_IMAGE_TO_GALLERY';

export const StagingAreaToolbarSaveSelectedToGalleryButton = memo(() => {
  const canvasManager = useCanvasManager();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const ctx = useCanvasSessionContext();
  const selectedItemOutputImageDTO = useStore(ctx.$selectedItemOutputImageDTO);
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

  const { t } = useTranslation();

  const saveSelectedImageToGallery = useCallback(async () => {
    if (!selectedItemOutputImageDTO) {
      return;
    }

    // To save the image to gallery, we will download it and re-upload it. This allows the user to delete the image
    // the gallery without borking the canvas, which may need this image to exist.
    const result = await withResultAsync(async () => {
      // Create a new file with the same name, which we will upload
      await copyImage(selectedItemOutputImageDTO.image_name, {
        // Image should show up in the Images tab
        image_category: 'general',
        is_intermediate: false,
        // TODO(psyche): Maybe this should just save to the currently-selected board?
        board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
        // We will do our own toast - opt out of the default handling
        silent: true,
      });
    });

    if (result.isOk()) {
      toast({
        id: TOAST_ID,
        title: t('controlLayers.savedToGalleryOk'),
        status: 'success',
      });
    } else {
      toast({
        id: TOAST_ID,
        title: t('controlLayers.savedToGalleryError'),
        status: 'error',
      });
    }
  }, [autoAddBoardId, selectedItemOutputImageDTO, t]);

  return (
    <IconButton
      tooltip={t('controlLayers.stagingArea.saveToGallery')}
      aria-label={t('controlLayers.stagingArea.saveToGallery')}
      icon={<PiFloppyDiskBold />}
      onClick={saveSelectedImageToGallery}
      colorScheme="invokeBlue"
      isDisabled={!selectedItemOutputImageDTO || !shouldShowStagedImage}
    />
  );
});

StagingAreaToolbarSaveSelectedToGalleryButton.displayName = 'StagingAreaToolbarSaveSelectedToGalleryButton';
