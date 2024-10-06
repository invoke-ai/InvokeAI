import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { imagesToChangeSelected, isModalOpenChanged } from 'features/changeBoardModal/store/slice';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFoldersBold } from 'react-icons/pi';

export const ImageMenuItemChangeBoard = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();

  const onClick = useCallback(() => {
    dispatch(imagesToChangeSelected([imageDTO]));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, imageDTO]);

  return (
    <MenuItem icon={<PiFoldersBold />} onClickCapture={onClick}>
      {t('boards.changeBoard')}
    </MenuItem>
  );
});

ImageMenuItemChangeBoard.displayName = 'ImageMenuItemChangeBoard';
