import { MenuDivider, MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imagesToChangeSelected, isModalOpenChanged } from 'features/changeBoardModal/store/slice';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo, useCallback } from 'react';
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
  const deleteImageModal = useDeleteImageModalApi();

  const isBulkDownloadEnabled = useFeatureStatus('bulkDownload');

  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();
  const [bulkDownload] = useBulkDownloadImagesMutation();

  const handleChangeBoard = useCallback(() => {
    dispatch(imagesToChangeSelected(selection.map((s) => s.id)));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, selection]);

  const handleDeleteSelection = useCallback(() => {
    deleteImageModal.delete(selection.map((s) => s.id));
  }, [deleteImageModal, selection]);

  const handleStarSelection = useCallback(() => {
    starImages({ image_names: selection.map((s) => s.id) });
  }, [starImages, selection]);

  const handleUnstarSelection = useCallback(() => {
    unstarImages({ image_names: selection.map((s) => s.id) });
  }, [unstarImages, selection]);

  const handleBulkDownload = useCallback(() => {
    bulkDownload({ image_names: selection.map((s) => s.id) });
  }, [selection, bulkDownload]);

  return (
    <>
      <MenuItem icon={customStarUi ? customStarUi.on.icon : <PiStarBold />} onClickCapture={handleUnstarSelection}>
        {customStarUi ? customStarUi.off.text : `Unstar All`}
      </MenuItem>
      <MenuItem icon={customStarUi ? customStarUi.on.icon : <PiStarFill />} onClickCapture={handleStarSelection}>
        {customStarUi ? customStarUi.on.text : `Star All`}
      </MenuItem>
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
