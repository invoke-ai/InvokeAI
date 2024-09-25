import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSelectedImage } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';
import { useAddImagesToBoardMutation, useChangeImageIsIntermediateMutation } from 'services/api/endpoints/images';

export const StagingAreaToolbarSaveSelectedToGalleryButton = memo(() => {
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const selectedImage = useAppSelector(selectSelectedImage);
  const [addImageToBoard] = useAddImagesToBoardMutation();
  const [changeIsImageIntermediate] = useChangeImageIsIntermediateMutation();

  const { t } = useTranslation();

  const saveSelectedImageToGallery = useCallback(async () => {
    if (!selectedImage) {
      return;
    }
    if (autoAddBoardId !== 'none') {
      await addImageToBoard({ imageDTOs: [selectedImage.imageDTO], board_id: autoAddBoardId }).unwrap();
      // The changeIsImageIntermediate request will use the board_id on this specific imageDTO object, so we need to
      // update it before making the request - else the optimistic board updates will get out of whack.
      changeIsImageIntermediate({
        imageDTO: { ...selectedImage.imageDTO, board_id: autoAddBoardId },
        is_intermediate: false,
      });
    } else {
      changeIsImageIntermediate({
        imageDTO: selectedImage.imageDTO,
        is_intermediate: false,
      });
    }
  }, [addImageToBoard, autoAddBoardId, changeIsImageIntermediate, selectedImage]);

  return (
    <IconButton
      tooltip={t('controlLayers.stagingArea.saveToGallery')}
      aria-label={t('controlLayers.stagingArea.saveToGallery')}
      icon={<PiFloppyDiskBold />}
      onClick={saveSelectedImageToGallery}
      colorScheme="invokeBlue"
      isDisabled={!selectedImage || !selectedImage.imageDTO.is_intermediate}
    />
  );
});

StagingAreaToolbarSaveSelectedToGalleryButton.displayName = 'StagingAreaToolbarSaveSelectedToGalleryButton';
