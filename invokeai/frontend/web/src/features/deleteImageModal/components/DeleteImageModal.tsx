import { ConfirmationAlertDialog, Divider, Flex, FormControl, FormLabel, Switch, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { imageDeletionConfirmed } from 'features/deleteImageModal/store/actions';
import { getImageUsage, selectImageUsage } from 'features/deleteImageModal/store/selectors';
import {
  imageDeletionCanceled,
  isModalOpenChanged,
  selectDeleteImageModalSlice,
} from 'features/deleteImageModal/store/slice';
import type { ImageUsage } from 'features/deleteImageModal/store/types';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { selectSystemSlice, setShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { some } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ImageUsageMessage from './ImageUsageMessage';

const selectImageUsages = createMemoizedSelector(
  [selectDeleteImageModalSlice, selectNodesSlice, selectCanvasSlice, selectImageUsage, selectUpscaleSlice],
  (deleteImageModal, nodes, canvas, imagesUsage, upscale) => {
    const { imagesToDelete } = deleteImageModal;

    const allImageUsage = (imagesToDelete ?? []).map(({ image_name }) =>
      getImageUsage(nodes, canvas, upscale, image_name)
    );

    const imageUsageSummary: ImageUsage = {
      isUpscaleImage: some(allImageUsage, (i) => i.isUpscaleImage),
      isRasterLayerImage: some(allImageUsage, (i) => i.isRasterLayerImage),
      isInpaintMaskImage: some(allImageUsage, (i) => i.isInpaintMaskImage),
      isRegionalGuidanceImage: some(allImageUsage, (i) => i.isRegionalGuidanceImage),
      isNodesImage: some(allImageUsage, (i) => i.isNodesImage),
      isControlLayerImage: some(allImageUsage, (i) => i.isControlLayerImage),
      isReferenceImage: some(allImageUsage, (i) => i.isReferenceImage),
    };

    return {
      imagesToDelete,
      imagesUsage,
      imageUsageSummary,
    };
  }
);

const selectShouldConfirmOnDelete = createSelector(selectSystemSlice, (system) => system.shouldConfirmOnDelete);
const selectIsModalOpen = createSelector(
  selectDeleteImageModalSlice,
  (deleteImageModal) => deleteImageModal.isModalOpen
);

const DeleteImageModal = () => {
  useAssertSingleton('DeleteImageModal');
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const shouldConfirmOnDelete = useAppSelector(selectShouldConfirmOnDelete);
  const isModalOpen = useAppSelector(selectIsModalOpen);
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
      cancelButtonText={t('common.cancel')}
      acceptButtonText={t('common.delete')}
      acceptCallback={handleDelete}
      useInert={false}
    >
      <Flex direction="column" gap={3}>
        <ImageUsageMessage imageUsage={imageUsageSummary} />
        <Divider />
        <Text>{t('gallery.deleteImagePermanent')}</Text>
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
