import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { imagesToChangeSelected, isModalOpenChanged } from 'features/changeBoardModal/store/slice';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFoldersBold } from 'react-icons/pi';
import { useBoardAccess } from 'services/api/hooks/useBoardAccess';
import { useSelectedBoard } from 'services/api/hooks/useSelectedBoard';

export const ContextMenuItemChangeBoard = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();
  const selectedBoard = useSelectedBoard();
  const { canWriteImages } = useBoardAccess(selectedBoard);

  const onClick = useCallback(() => {
    dispatch(imagesToChangeSelected([imageDTO.image_name]));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, imageDTO]);

  return (
    <MenuItem icon={<PiFoldersBold />} onClickCapture={onClick} isDisabled={!canWriteImages}>
      {t('boards.changeBoard')}
    </MenuItem>
  );
});

ContextMenuItemChangeBoard.displayName = 'ContextMenuItemChangeBoard';
