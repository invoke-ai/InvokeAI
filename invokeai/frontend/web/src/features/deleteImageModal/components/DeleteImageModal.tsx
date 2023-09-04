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
import { some } from 'lodash-es';
import { ChangeEvent, memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { imageDeletionConfirmed } from '../store/actions';
import { getImageUsage, selectImageUsage } from '../store/selectors';
import { imageDeletionCanceled, isModalOpenChanged } from '../store/slice';
import ImageUsageMessage from './ImageUsageMessage';
import { ImageUsage } from '../store/types';

const selector = createSelector(
  [stateSelector, selectImageUsage],
  (state, imagesUsage) => {
    const { system, config, deleteImageModal } = state;
    const { shouldConfirmOnDelete } = system;
    const { canRestoreDeletedImagesFromBin } = config;
    const { imagesToDelete, isModalOpen } = deleteImageModal;

    const allImageUsage = (imagesToDelete ?? []).map(({ image_name }) =>
      getImageUsage(state, image_name)
    );

    const imageUsageSummary: ImageUsage = {
      isInitialImage: some(allImageUsage, (i) => i.isInitialImage),
      isCanvasImage: some(allImageUsage, (i) => i.isCanvasImage),
      isNodesImage: some(allImageUsage, (i) => i.isNodesImage),
      isControlNetImage: some(allImageUsage, (i) => i.isControlNetImage),
    };

    return {
      shouldConfirmOnDelete,
      canRestoreDeletedImagesFromBin,
      imagesToDelete,
      imagesUsage,
      isModalOpen,
      imageUsageSummary,
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
    imagesToDelete,
    imagesUsage,
    isModalOpen,
    imageUsageSummary,
  } = useAppSelector(selector);

  const handleChangeShouldConfirmOnDelete = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldConfirmOnDelete(!e.target.checked)),
    [dispatch]
  );

  const handleClose = useCallback(() => {
    dispatch(imageDeletionCanceled());
    dispatch(isModalOpenChanged(false));
  }, [dispatch]);

  const handleDelete = useCallback(() => {
    if (!imagesToDelete.length || !imagesUsage.length) {
      return;
    }
    dispatch(imageDeletionCanceled());
    dispatch(
      imageDeletionConfirmed({ imageDTOs: imagesToDelete, imagesUsage })
    );
  }, [dispatch, imagesToDelete, imagesUsage]);

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
              <ImageUsageMessage imageUsage={imageUsageSummary} />
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
