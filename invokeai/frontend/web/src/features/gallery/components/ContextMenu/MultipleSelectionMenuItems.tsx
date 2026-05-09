import { MenuDivider, MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imagesToChangeSelected, isModalOpenChanged } from 'features/changeBoardModal/store/slice';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold, PiFoldersBold, PiStarBold, PiStarFill, PiTrashSimpleBold } from 'react-icons/pi';
import {
  useBulkDownloadImagesMutation,
  useStarImagesMutation,
  useUnstarImagesMutation,
} from 'services/api/endpoints/images';
import { useBoardAccess } from 'services/api/hooks/useBoardAccess';
import { useSelectedBoard } from 'services/api/hooks/useSelectedBoard';

const MultipleSelectionMenuItems = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selection = useAppSelector((s) => s.gallery.selection);
  const deleteImageModal = useDeleteImageModalApi();
  const selectedBoard = useSelectedBoard();
  const { canWriteImages } = useBoardAccess(selectedBoard);

  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();
  const [bulkDownload] = useBulkDownloadImagesMutation();

  const handleChangeBoard = useCallback(() => {
    dispatch(imagesToChangeSelected(selection));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, selection]);

  const handleDeleteSelection = useCallback(() => {
    deleteImageModal.delete(selection);
  }, [deleteImageModal, selection]);

  const handleStarSelection = useCallback(() => {
    starImages({ image_names: selection });
  }, [starImages, selection]);

  const handleUnstarSelection = useCallback(() => {
    unstarImages({ image_names: selection });
  }, [unstarImages, selection]);

  const handleBulkDownload = useCallback(() => {
    bulkDownload({ image_names: selection });
  }, [selection, bulkDownload]);

  return (
    <>
      <MenuItem icon={<PiStarBold />} onClickCapture={handleUnstarSelection}>
        Unstar All
      </MenuItem>
      <MenuItem icon={<PiStarFill />} onClickCapture={handleStarSelection}>
        Star All
      </MenuItem>
      <MenuItem icon={<PiDownloadSimpleBold />} onClickCapture={handleBulkDownload}>
        {t('gallery.downloadSelection')}
      </MenuItem>
      <MenuItem icon={<PiFoldersBold />} onClickCapture={handleChangeBoard} isDisabled={!canWriteImages}>
        {t('boards.changeBoard')}
      </MenuItem>
      <MenuDivider />
      <MenuItem
        color="error.300"
        icon={<PiTrashSimpleBold />}
        onClickCapture={handleDeleteSelection}
        isDisabled={!canWriteImages}
      >
        {t('gallery.deleteSelection')}
      </MenuItem>
    </>
  );
};

export default memo(MultipleSelectionMenuItems);
