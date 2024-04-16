import { ConfirmationAlertDialog, Divider, Flex, FormControl, FormLabel, Switch, Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { selectControlAdaptersSlice } from 'features/controlAdapters/store/controlAdaptersSlice';
import { imageDeletionConfirmed } from 'features/deleteImageModal/store/actions';
import { getImageUsage, selectImageUsage } from 'features/deleteImageModal/store/selectors';
import {
  imageDeletionCanceled,
  isModalOpenChanged,
  selectDeleteImageModalSlice,
} from 'features/deleteImageModal/store/slice';
import type { ImageUsage } from 'features/deleteImageModal/store/types';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { setShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { some } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ImageUsageMessage from './ImageUsageMessage';

const selectImageUsages = createMemoizedSelector(
  [
    selectDeleteImageModalSlice,
    selectGenerationSlice,
    selectCanvasSlice,
    selectNodesSlice,
    selectControlAdaptersSlice,
    selectImageUsage,
  ],
  (deleteImageModal, generation, canvas, nodes, controlAdapters, imagesUsage) => {
    const { imagesToDelete } = deleteImageModal;

    const allImageUsage = (imagesToDelete ?? []).map(({ image_name }) =>
      getImageUsage(generation, canvas, nodes, controlAdapters, image_name)
    );

    const imageUsageSummary: ImageUsage = {
      isInitialImage: some(allImageUsage, (i) => i.isInitialImage),
      isCanvasImage: some(allImageUsage, (i) => i.isCanvasImage),
      isNodesImage: some(allImageUsage, (i) => i.isNodesImage),
      isControlImage: some(allImageUsage, (i) => i.isControlImage),
    };

    return {
      imagesToDelete,
      imagesUsage,
      imageUsageSummary,
    };
  }
);

const DeleteImageModal = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const shouldConfirmOnDelete = useAppSelector((s) => s.system.shouldConfirmOnDelete);
  const canRestoreDeletedImagesFromBin = useAppSelector((s) => s.config.canRestoreDeletedImagesFromBin);
  const isModalOpen = useAppSelector((s) => s.deleteImageModal.isModalOpen);
  const { imagesToDelete, imagesUsage, imageUsageSummary } = useAppSelector(selectImageUsages);

  const handleChangeShouldConfirmOnDelete = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldConfirmOnDelete(!e.target.checked)),
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
    dispatch(imageDeletionConfirmed({ imageDTOs: imagesToDelete, imagesUsage }));
  }, [dispatch, imagesToDelete, imagesUsage]);

  return (
    <ConfirmationAlertDialog
      title={t('gallery.deleteImage', { count: imagesToDelete.length })}
      isOpen={isModalOpen}
      onClose={handleClose}
      cancelButtonText={t('boards.cancel')}
      acceptButtonText={t('controlnet.delete')}
      acceptCallback={handleDelete}
    >
      <Flex direction="column" gap={3}>
        <ImageUsageMessage imageUsage={imageUsageSummary} />
        <Divider />
        <Text>{canRestoreDeletedImagesFromBin ? t('gallery.deleteImageBin') : t('gallery.deleteImagePermanent')}</Text>
        <Text>{t('common.areYouSure')}</Text>
        <FormControl>
          <FormLabel>{t('common.dontAskMeAgain')}</FormLabel>
          <Switch isChecked={!shouldConfirmOnDelete} onChange={handleChangeShouldConfirmOnDelete} />
        </FormControl>
      </Flex>
    </ConfirmationAlertDialog>
  );
};

export default memo(DeleteImageModal);
