import { MenuItem } from '@invoke-ai/ui-library';
import { useChangeBoardModalApi } from 'features/changeBoardModal/store/state';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFoldersBold } from 'react-icons/pi';
import { isImageDTO, isVideoDTO } from 'services/api/types';

export const ContextMenuItemChangeBoard = memo(() => {
  const { t } = useTranslation();
  const itemDTO = useItemDTOContext();
  const changeBoardModal = useChangeBoardModalApi();

  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      changeBoardModal.openWithImages([itemDTO.image_name]);
    } else if (isVideoDTO(itemDTO)) {
      changeBoardModal.openWithVideos([itemDTO.video_id]);
    }
  }, [changeBoardModal, itemDTO]);

  return (
    <MenuItem icon={<PiFoldersBold />} onClickCapture={onClick}>
      {t('boards.changeBoard')}
    </MenuItem>
  );
});

ContextMenuItemChangeBoard.displayName = 'ContextMenuItemChangeBoard';
