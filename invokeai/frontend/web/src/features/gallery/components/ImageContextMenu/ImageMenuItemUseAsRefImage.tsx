import { MenuItem } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { getDefaultRefImageConfig } from 'features/controlLayers/hooks/addLayerHooks';
import { refImageAdded } from 'features/controlLayers/store/refImagesSlice';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold } from 'react-icons/pi';

export const ImageMenuItemUseAsRefImage = memo(() => {
  const { t } = useTranslation();
  const store = useAppStore();
  const imageDTO = useImageDTOContext();
  const imageViewer = useImageViewer();

  const onClickNewGlobalReferenceImageFromImage = useCallback(() => {
    const { dispatch, getState } = store;
    const config = getDefaultRefImageConfig(getState);
    config.image = imageDTOToImageWithDims(imageDTO);
    dispatch(refImageAdded({ overrides: { config } }));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, imageViewer, store, t]);

  return (
    <MenuItem icon={<PiImageBold />} onClickCapture={onClickNewGlobalReferenceImageFromImage}>
      Use as Reference Image
    </MenuItem>
  );
});

ImageMenuItemUseAsRefImage.displayName = 'ImageMenuItemUseAsRefImage';
