import { MenuItem } from '@invoke-ai/ui-library';
import { useVideoDTOContext } from 'features/gallery/contexts/VideoDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold } from 'react-icons/pi';

export const ContextMenuItemOpenInNewTabVideo = memo(() => {
  const { t } = useTranslation();
  const videoDTO = useVideoDTOContext();
  const onClick = useCallback(() => {
    window.open(videoDTO.video_url, '_blank', 'noopener,noreferrer');
  }, [videoDTO.video_url]);

  return (
    <MenuItem icon={<PiArrowSquareOutBold />} onClickCapture={onClick}>
      {t('common.openInNewTab')}
    </MenuItem>
  );
});

ContextMenuItemOpenInNewTabVideo.displayName = 'ContextMenuItemOpenInNewTabVideo';
