import { MenuItem } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/storeHooks';
import { getDefaultRefImageConfig } from 'features/controlLayers/hooks/addLayerHooks';
import { refImageAdded } from 'features/controlLayers/store/refImagesSlice';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { useItemDTOContext, useItemDTOContextImageOnly } from 'features/gallery/contexts/ItemDTOContext';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

export const ContextMenuItemUseAsRefImage = memo(() => {
  const { t } = useTranslation();
  const store = useAppStore();
  const imageDTO = useItemDTOContextImageOnly();

  const onClickNewGlobalReferenceImageFromImage = useCallback(() => {
    const { dispatch, getState } = store;
    const config = getDefaultRefImageConfig(getState);
    config.image = imageDTOToImageWithDims(imageDTO);
    dispatch(refImageAdded({ overrides: { config } }));
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, store, t]);

  return (
    <MenuItem icon={<PiImageBold />} onClickCapture={onClickNewGlobalReferenceImageFromImage}>
      {t('controlLayers.useAsReferenceImage')}
    </MenuItem>
  );
});

  ContextMenuItemUseAsRefImage.displayName = 'ContextMenuItemUseAsRefImage';
