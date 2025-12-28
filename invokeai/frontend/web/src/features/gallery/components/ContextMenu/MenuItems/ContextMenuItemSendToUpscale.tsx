import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShareFatBold } from 'react-icons/pi';

export const ContextMenuItemSendToUpscale = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();

  const handleSendToCanvas = useCallback(() => {
    dispatch(upscaleInitialImageChanged(imageDTOToImageWithDims(imageDTO)));
    navigationApi.switchToTab('upscaling');
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToUpscale'),
      status: 'success',
    });
  }, [dispatch, imageDTO, t]);

  return (
    <MenuItem icon={<PiShareFatBold />} onClickCapture={handleSendToCanvas} id="send-to-upscale">
      {t('parameters.sendToUpscale')}
    </MenuItem>
  );
});

ContextMenuItemSendToUpscale.displayName = 'ContextMenuItemSendToUpscale';
