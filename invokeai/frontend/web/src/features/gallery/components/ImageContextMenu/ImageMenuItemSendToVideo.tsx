import { MenuItem } from '@invoke-ai/ui-library';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { setCurrentVideo } from 'features/ui/layouts/video-store';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiVideoBold } from 'react-icons/pi';

export const ImageMenuItemSendToVideo = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();

  const onClick = useCallback(() => {
    // For now, we'll use the image URL as a video source
    // In a real implementation, you might want to convert the image to video or use a different approach
    setCurrentVideo(imageDTO.image_url);
    navigationApi.switchToTab('video');
  }, [imageDTO.image_url]);

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