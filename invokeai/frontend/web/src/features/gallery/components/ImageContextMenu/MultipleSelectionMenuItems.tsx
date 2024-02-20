import { MenuDivider, MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imagesToChangeSelected, isModalOpenChanged } from 'features/changeBoardModal/store/slice';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold, PiFoldersBold, PiStarBold, PiStarFill, PiTrashSimpleBold } from 'react-icons/pi';
import {
  useBulkDownloadImagesMutation,
  useStarImagesMutation,
  useUnstarImagesMutation,
} from 'services/api/endpoints/images';

const MultipleSelectionMenuItems = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selection = useAppSelector((s) => s.gallery.selection);
  const customStarUi = useStore($customStarUI);

  const isBulkDownloadEnabled = useFeatureStatus('bulkDownload').isFeatureEnabled;

  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();
  const [bulkDownload] = useBulkDownloadImagesMutation();

  const handleChangeBoard = useCallback(() => {
    dispatch(imagesToChangeSelected(selection));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, selection]);

  const handleDeleteSelection = useCallback(() => {
    dispatch(imagesToDeleteSelected(selection));
  }, [dispatch, selection]);

  const handleStarSelection = useCallback(() => {
    starImages({ imageDTOs: selection });
  }, [starImages, selection]);

  const handleUnstarSelection = useCallback(() => {
    unstarImages({ imageDTOs: selection });
  }, [unstarImages, selection]);

  const handleBulkDownload = useCallback(() => {
    bulkDownload({ image_names: selection.map((img) => img.image_name) });
  }, [selection, bulkDownload]);

  const areAllStarred = useMemo(() => {
    return selection.every((img) => img.starred);
  }, [selection]);

  const areAllUnstarred = useMemo(() => {
    return selection.every((img) => !img.starred);
  }, [selection]);

  return (
    <>
      {areAllStarred && (
        <MenuItem icon={customStarUi ? customStarUi.on.icon : <PiStarBold />} onClickCapture={handleUnstarSelection}>
          {customStarUi ? customStarUi.off.text : `Unstar All`}
        </MenuItem>
      )}
      {(areAllUnstarred || (!areAllStarred && !areAllUnstarred)) && (
        <MenuItem icon={customStarUi ? customStarUi.on.icon : <PiStarFill />} onClickCapture={handleStarSelection}>
          {customStarUi ? customStarUi.on.text : `Star All`}
        </MenuItem>
      )}
      {isBulkDownloadEnabled && (
        <MenuItem icon={<PiDownloadSimpleBold />} onClickCapture={handleBulkDownload}>
          {t('gallery.downloadSelection')}
        </MenuItem>
      )}
      <MenuItem icon={<PiFoldersBold />} onClickCapture={handleChangeBoard}>
        {t('boards.changeBoard')}
      </MenuItem>
      <MenuDivider />
      <MenuItem color="error.300" icon={<PiTrashSimpleBold />} onClickCapture={handleDeleteSelection}>
        {t('gallery.deleteSelection')}
      </MenuItem>
    </>
  );
};

export default memo(MultipleSelectionMenuItems);
