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
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAISwitch from 'common/components/IAISwitch';
import { imageDeletionConfirmed } from 'features/deleteImageModal/store/actions';
import {
  getImageUsage,
  selectImageUsage,
} from 'features/deleteImageModal/store/selectors';
import {
  imageDeletionCanceled,
  isModalOpenChanged,
} from 'features/deleteImageModal/store/slice';
import { ImageUsage } from 'features/deleteImageModal/store/types';
import { setShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { some } from 'lodash-es';
import { ChangeEvent, memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import ImageUsageMessage from './ImageUsageMessage';

const selector = createMemoizedSelector(
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
      isControlImage: some(allImageUsage, (i) => i.isControlImage),
    };

    return {
      shouldConfirmOnDelete,
      canRestoreDeletedImagesFromBin,
      imagesToDelete,
      imagesUsage,
      isModalOpen,
      imageUsageSummary,
    };
  }
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
              {t('boards.cancel')}
            </IAIButton>
            <IAIButton colorScheme="error" onClick={handleDelete} ml={3}>
              {t('controlnet.delete')}
            </IAIButton>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(DeleteImageModal);
