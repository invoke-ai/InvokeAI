import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Divider,
  Flex,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton from 'common/components/IAIButton';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldConfirmOnDelete } from 'features/system/store/systemSlice';

import { stateSelector } from 'app/store/store';
import { ChangeEvent, memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { imageDeletionConfirmed } from '../store/actions';
import { selectImageUsage } from '../store/imageDeletionSelectors';
import {
  imageToDeleteCleared,
  isModalOpenChanged,
} from '../store/imageDeletionSlice';
import ImageUsageMessage from './ImageUsageMessage';

const selector = createSelector(
  [stateSelector, selectImageUsage],
  ({ system, config, imageDeletion }, imageUsage) => {
    const { shouldConfirmOnDelete } = system;
    const { canRestoreDeletedImagesFromBin } = config;
    const { imageToDelete, isModalOpen } = imageDeletion;
    return {
      shouldConfirmOnDelete,
      canRestoreDeletedImagesFromBin,
      imageToDelete,
      imageUsage,
      isModalOpen,
    };
  },
  defaultSelectorOptions
);

const DeleteImageModal = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const {
    shouldConfirmOnDelete,
    canRestoreDeletedImagesFromBin,
    imageToDelete,
    imageUsage,
    isModalOpen,
  } = useAppSelector(selector);

  const handleChangeShouldConfirmOnDelete = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldConfirmOnDelete(!e.target.checked)),
    [dispatch]
  );

  const handleClose = useCallback(() => {
    dispatch(imageToDeleteCleared());
    dispatch(isModalOpenChanged(false));
  }, [dispatch]);

  const handleDelete = useCallback(() => {
    if (!imageToDelete || !imageUsage) {
      return;
    }
    dispatch(imageToDeleteCleared());
    dispatch(imageDeletionConfirmed({ imageDTO: imageToDelete, imageUsage }));
  }, [dispatch, imageToDelete, imageUsage]);

  const cancelRef = useRef<HTMLButtonElement>(null);

  return (
    <AlertDialog
      isOpen={isModalOpen}
      onClose={handleClose}
      leastDestructiveRef={cancelRef}
      isCentered
    >
      <AlertDialogOverlay>
        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            {t('gallery.deleteImage')}
          </AlertDialogHeader>

          <AlertDialogBody>
            <Flex direction="column" gap={3}>
              <ImageUsageMessage imageUsage={imageUsage} />
              <Divider />
              <Text>
                {canRestoreDeletedImagesFromBin
                  ? t('gallery.deleteImageBin')
                  : t('gallery.deleteImagePermanent')}
              </Text>
              <Text>{t('common.areYouSure')}</Text>
              <IAISwitch
                label={t('common.dontAskMeAgain')}
                isChecked={!shouldConfirmOnDelete}
                onChange={handleChangeShouldConfirmOnDelete}
              />
            </Flex>
          </AlertDialogBody>
          <AlertDialogFooter>
            <IAIButton ref={cancelRef} onClick={handleClose}>
              Cancel
            </IAIButton>
            <IAIButton colorScheme="error" onClick={handleDelete} ml={3}>
              Delete
            </IAIButton>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(DeleteImageModal);
