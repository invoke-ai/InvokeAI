import { MenuItem } from '@invoke-ai/ui-library';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { selectVideoFirstFrameImage, videoFirstFrameImageChanged } from 'features/parameters/store/videoSlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiVideoBold } from 'react-icons/pi';
import { useDispatch } from 'react-redux';

export const ImageMenuItemSendToVideo = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const dispatch = useDispatch();

  const onClick = useCallback(() => {
    dispatch(videoFirstFrameImageChanged(imageDTO));
    navigationApi.switchToTab('video');
  }, [imageDTO]);

  return (
    <MenuItem
      icon={<PiVideoBold />}
      onClickCapture={onClick}
      aria-label={"Send to Video"}
    >
      Send to Video
    </MenuItem>
  );
});

ImageMenuItemSendToVideo.displayName = 'ImageMenuItemSendToVideo'; 