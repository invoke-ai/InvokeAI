import { MenuDivider, MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imagesToChangeSelected, isModalOpenChanged } from 'features/changeBoardModal/store/slice';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { selectSelection } from 'features/gallery/store/gallerySelectors';
import { isVideoName } from 'features/gallery/store/types';
import { memo, useCallback, useMemo } from 'react';
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
  const selection = useAppSelector(selectSelection);
  const deleteImageModal = useDeleteImageModalApi();
  const selectedBoard = useSelectedBoard();
  const { canWriteImages } = useBoardAccess(selectedBoard);

  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();
  const [bulkDownload] = useBulkDownloadImagesMutation();

  // The gallery selection can contain mixed image+video names. Each menu only acts on its
  // own kind so the action is unambiguous; right-clicking a video surfaces the video
  // equivalent of this menu.
  const imageNames = useMemo(() => selection.filter((name) => !isVideoName(name)), [selection]);
  const count = imageNames.length;
  const hasImages = count > 0;

  const handleChangeBoard = useCallback(() => {
    dispatch(imagesToChangeSelected(imageNames));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, imageNames]);

  const handleDeleteSelection = useCallback(() => {
    deleteImageModal.delete(imageNames);
  }, [deleteImageModal, imageNames]);

  const handleStarSelection = useCallback(() => {
    starImages({ image_names: imageNames });
  }, [starImages, imageNames]);

  const handleUnstarSelection = useCallback(() => {
    unstarImages({ image_names: imageNames });
  }, [unstarImages, imageNames]);

  const handleBulkDownload = useCallback(() => {
    bulkDownload({ image_names: imageNames });
  }, [imageNames, bulkDownload]);

  return (
    <>
      <MenuItem icon={<PiStarBold />} onClickCapture={handleUnstarSelection} isDisabled={!hasImages}>
        {t('gallery.unstarImage', { count })}
      </MenuItem>
      <MenuItem icon={<PiStarFill />} onClickCapture={handleStarSelection} isDisabled={!hasImages}>
        {t('gallery.starImage', { count })}
      </MenuItem>
      <MenuItem icon={<PiDownloadSimpleBold />} onClickCapture={handleBulkDownload} isDisabled={!hasImages}>
        {t('gallery.downloadImage', { count })}
      </MenuItem>
      <MenuItem icon={<PiFoldersBold />} onClickCapture={handleChangeBoard} isDisabled={!hasImages || !canWriteImages}>
        {t('boards.changeBoardImage', { count })}
      </MenuItem>
      <MenuDivider />
      <MenuItem
        color="error.300"
        icon={<PiTrashSimpleBold />}
        onClickCapture={handleDeleteSelection}
        isDisabled={!hasImages || !canWriteImages}
      >
        {t('gallery.deleteImage', { count })}
      </MenuItem>
    </>
  );
};

export default memo(MultipleSelectionMenuItems);
