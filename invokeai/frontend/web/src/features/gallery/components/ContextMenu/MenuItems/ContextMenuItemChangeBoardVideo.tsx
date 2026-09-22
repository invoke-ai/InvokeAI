import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { isModalOpenChanged, videosToChangeSelected } from 'features/changeBoardModal/store/slice';
import { useVideoDTOContext } from 'features/gallery/contexts/VideoDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFoldersBold } from 'react-icons/pi';

export const ContextMenuItemChangeBoardVideo = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const videoDTO = useVideoDTOContext();

  const onClick = useCallback(() => {
    dispatch(videosToChangeSelected([videoDTO.video_name]));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, videoDTO.video_name]);

  return (
    <MenuItem icon={<PiFoldersBold />} onClickCapture={onClick}>
      {t('boards.changeBoard')}
    </MenuItem>
  );
});

ContextMenuItemChangeBoardVideo.displayName = 'ContextMenuItemChangeBoardVideo';
