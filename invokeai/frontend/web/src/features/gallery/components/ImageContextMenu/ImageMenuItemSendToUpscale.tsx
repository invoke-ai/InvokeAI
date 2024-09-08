import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShareFatBold } from 'react-icons/pi';

export const ImageMenuItemSendToUpscale = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();

  const handleSendToCanvas = useCallback(() => {
    dispatch(upscaleInitialImageChanged(imageDTO));
    dispatch(setActiveTab('upscaling'));
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

ImageMenuItemSendToUpscale.displayName = 'ImageMenuItemSendToUpscale';
