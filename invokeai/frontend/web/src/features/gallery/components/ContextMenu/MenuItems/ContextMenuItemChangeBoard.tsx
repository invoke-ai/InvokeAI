import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { imagesToChangeSelected, isModalOpenChanged } from 'features/changeBoardModal/store/slice';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFoldersBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemChangeBoard = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const itemDTO = useItemDTOContext();

  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      dispatch(imagesToChangeSelected([itemDTO.image_name]));
    } else {
      // dispatch(videosToChangeSelected([itemDTO.video_id]));
    }
    dispatch(isModalOpenChanged(true));
  }, [dispatch, itemDTO]);

  return (
    <MenuItem icon={<PiFoldersBold />} onClickCapture={onClick}>
      {t('boards.changeBoard')}
    </MenuItem>
  );
});

ContextMenuItemChangeBoard.displayName = 'ContextMenuItemChangeBoard';
