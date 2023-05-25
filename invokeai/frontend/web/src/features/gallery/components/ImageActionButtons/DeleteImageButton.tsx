import { createSelector } from '@reduxjs/toolkit';

import { useDisclosure } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { systemSelector } from 'features/system/store/systemSelectors';

import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';
import { memo, useCallback } from 'react';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import DeleteImageModal from '../DeleteImageModal';
import { requestedImageDeletion } from 'features/gallery/store/actions';
import { ImageDTO } from 'services/api';

const selector = createSelector(
  [systemSelector],
  (system) => {
    const { isProcessing, isConnected, shouldConfirmOnDelete } = system;

    return {
      canDeleteImage: isConnected && !isProcessing,
      shouldConfirmOnDelete,
      isProcessing,
      isConnected,
    };
  },
  defaultSelectorOptions
);

type DeleteImageButtonProps = {
  image: ImageDTO | undefined;
};

const DeleteImageButton = (props: DeleteImageButtonProps) => {
  const { image } = props;
  const dispatch = useAppDispatch();
  const { isProcessing, isConnected, canDeleteImage, shouldConfirmOnDelete } =
    useAppSelector(selector);

  const {
    isOpen: isDeleteDialogOpen,
    onOpen: onDeleteDialogOpen,
    onClose: onDeleteDialogClose,
  } = useDisclosure();

  const { t } = useTranslation();

  const handleDelete = useCallback(() => {
    if (canDeleteImage && image) {
      dispatch(requestedImageDeletion(image));
    }
  }, [image, canDeleteImage, dispatch]);

  const handleInitiateDelete = useCallback(() => {
    if (shouldConfirmOnDelete) {
      onDeleteDialogOpen();
    } else {
      handleDelete();
    }
  }, [shouldConfirmOnDelete, onDeleteDialogOpen, handleDelete]);

  useHotkeys('delete', handleInitiateDelete, [
    image,
    shouldConfirmOnDelete,
    isConnected,
    isProcessing,
  ]);

  return (
    <>
      <IAIIconButton
        onClick={handleInitiateDelete}
        icon={<FaTrash />}
        tooltip={`${t('gallery.deleteImage')} (Del)`}
        aria-label={`${t('gallery.deleteImage')} (Del)`}
        isDisabled={!image || !isConnected}
        colorScheme="error"
      />
      {image && (
        <DeleteImageModal
          isOpen={isDeleteDialogOpen}
          onClose={onDeleteDialogClose}
          handleDelete={handleDelete}
        />
      )}
    </>
  );
};

export default memo(DeleteImageButton);
