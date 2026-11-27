import type { ContextMenuProps } from '@invoke-ai/ui-library';
import { ContextMenu, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectAutoAddBoardId, selectAutoAssignBoardOnClick } from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadBold, PiPlusBold, PiTrashSimpleBold } from 'react-icons/pi';
import { useBulkDownloadImagesMutation } from 'services/api/endpoints/images';

import { $boardToDelete } from './DeleteBoardModal';

type Props = {
  children: ContextMenuProps<HTMLDivElement>['children'];
};

const selectIsSelectedForAutoAdd = createSelector(selectAutoAddBoardId, (autoAddBoardId) => autoAddBoardId === 'none');

const NoBoardBoardContextMenu = ({ children }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoAssignBoardOnClick = useAppSelector(selectAutoAssignBoardOnClick);
  const isSelectedForAutoAdd = useAppSelector(selectIsSelectedForAutoAdd);

  const [bulkDownload] = useBulkDownloadImagesMutation();

  const handleSetAutoAdd = useCallback(() => {
    dispatch(autoAddBoardIdChanged('none'));
  }, [dispatch]);

  const handleBulkDownload = useCallback(() => {
    bulkDownload({ image_names: [], board_id: 'none' });
  }, [bulkDownload]);

  const setUncategorizedImagesAsToBeDeleted = useCallback(() => {
    $boardToDelete.set('none');
  }, []);

  const renderMenuFunc = useCallback(
    () => (
      <MenuList visibility="visible">
        <MenuGroup title={t('boards.uncategorized')}>
          {!autoAssignBoardOnClick && (
            <MenuItem icon={<PiPlusBold />} isDisabled={isSelectedForAutoAdd} onClick={handleSetAutoAdd}>
              {isSelectedForAutoAdd ? t('boards.selectedForAutoAdd') : t('boards.menuItemAutoAdd')}
            </MenuItem>
          )}
          <MenuItem icon={<PiDownloadBold />} onClickCapture={handleBulkDownload}>
            {t('boards.downloadBoard')}
          </MenuItem>
          <MenuItem
            color="error.300"
            icon={<PiTrashSimpleBold />}
            onClick={setUncategorizedImagesAsToBeDeleted}
            isDestructive
          >
            {t('boards.deleteAllUncategorizedImages')}
          </MenuItem>
        </MenuGroup>
      </MenuList>
    ),
    [
      autoAssignBoardOnClick,
      handleBulkDownload,
      handleSetAutoAdd,
      isSelectedForAutoAdd,
      t,
      setUncategorizedImagesAsToBeDeleted,
    ]
  );

  return <ContextMenu renderMenu={renderMenuFunc}>{children}</ContextMenu>;
};

export default memo(NoBoardBoardContextMenu);
