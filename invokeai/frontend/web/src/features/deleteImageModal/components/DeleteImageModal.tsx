import { Divider, Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvConfirmationAlertDialog } from 'common/components/InvConfirmationAlertDialog/InvConfirmationAlertDialog';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import { InvText } from 'common/components/InvText/wrapper';
import { imageDeletionConfirmed } from 'features/deleteImageModal/store/actions';
import {
  getImageUsage,
  selectImageUsage,
} from 'features/deleteImageModal/store/selectors';
import {
  imageDeletionCanceled,
  isModalOpenChanged,
} from 'features/deleteImageModal/store/slice';
import type { ImageUsage } from 'features/deleteImageModal/store/types';
import { setShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { some } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
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

  return (
    <InvConfirmationAlertDialog
      title={t('gallery.deleteImage')}
      isOpen={isModalOpen}
      onClose={handleClose}
      cancelButtonText={t('boards.cancel')}
      acceptButtonText={t('controlnet.delete')}
      acceptCallback={handleDelete}
    >
      <Flex direction="column" gap={3}>
        <ImageUsageMessage imageUsage={imageUsageSummary} />
        <Divider />
        <InvText>
          {canRestoreDeletedImagesFromBin
            ? t('gallery.deleteImageBin')
            : t('gallery.deleteImagePermanent')}
        </InvText>
        <InvText>{t('common.areYouSure')}</InvText>
        <InvControl label={t('common.dontAskMeAgain')}>
          <InvSwitch
            isChecked={!shouldConfirmOnDelete}
            onChange={handleChangeShouldConfirmOnDelete}
          />
        </InvControl>
      </Flex>
    </InvConfirmationAlertDialog>
  );
};

export default memo(DeleteImageModal);
