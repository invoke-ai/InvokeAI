import { MenuItem } from '@invoke-ai/ui-library';
import { useItemDTOContext, useItemDTOContextImageOnly } from 'features/gallery/contexts/ItemDTOContext';
import {  startingFrameImageChanged } from 'features/parameters/store/videoSlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiVideoBold } from 'react-icons/pi';
import { useDispatch } from 'react-redux';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemSendToVideo = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useItemDTOContextImageOnly();
  const dispatch = useDispatch();

  const onClick = useCallback(() => {
    dispatch(startingFrameImageChanged(imageDTO));
    navigationApi.switchToTab('video');
  }, [imageDTO, dispatch]);

  return (
    <MenuItem
      icon={<PiVideoBold />}
      onClickCapture={onClick}
      aria-label="Send to Video"
    >
      Send to Video
    </MenuItem>
  );
});

ContextMenuItemSendToVideo.displayName = 'ContextMenuItemSendToVideo'; 