import type { ContextMenuProps } from '@invoke-ai/ui-library';
import { ContextMenu, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadBold, PiPlusBold } from 'react-icons/pi';
import { useBulkDownloadImagesMutation } from 'services/api/endpoints/images';

type Props = {
  children: ContextMenuProps<HTMLDivElement>['children'];
};

const NoBoardBoardContextMenu = ({ children }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoAssignBoardOnClick = useAppSelector((s) => s.gallery.autoAssignBoardOnClick);
  const isSelectedForAutoAdd = useAppSelector((s) => s.gallery.autoAddBoardId === 'none');
  const isBulkDownloadEnabled = useFeatureStatus('bulkDownload');

  const [bulkDownload] = useBulkDownloadImagesMutation();

  const handleSetAutoAdd = useCallback(() => {
    dispatch(autoAddBoardIdChanged('none'));
  }, [dispatch]);

  const handleBulkDownload = useCallback(() => {
    bulkDownload({ image_names: [], board_id: 'none' });
  }, [bulkDownload]);

  const renderMenuFunc = useCallback(
    () => (
      <MenuList visibility="visible">
        <MenuGroup title={t('boards.uncategorized')}>
          <MenuItem
            icon={<PiPlusBold />}
            isDisabled={isSelectedForAutoAdd || autoAssignBoardOnClick}
            onClick={handleSetAutoAdd}
          >
            {t('boards.menuItemAutoAdd')}
          </MenuItem>
          {isBulkDownloadEnabled && (
            <MenuItem icon={<PiDownloadBold />} onClickCapture={handleBulkDownload}>
              {t('boards.downloadBoard')}
            </MenuItem>
          )}
        </MenuGroup>
      </MenuList>
    ),
    [autoAssignBoardOnClick, handleBulkDownload, handleSetAutoAdd, isBulkDownloadEnabled, isSelectedForAutoAdd, t]
  );

  return <ContextMenu renderMenu={renderMenuFunc}>{children}</ContextMenu>;
};

export default memo(NoBoardBoardContextMenu);
