import { MenuItem } from '@chakra-ui/react';
import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  imagesToChangeSelected,
  isModalOpenChanged,
} from 'features/changeBoardModal/store/slice';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { memo, useCallback, useMemo } from 'react';
import { FaDownload, FaFolder, FaTrash } from 'react-icons/fa';
import { MdStar, MdStarBorder } from 'react-icons/md';
import {
  useBulkDownloadImagesMutation,
  useStarImagesMutation,
  useUnstarImagesMutation,
} from 'services/api/endpoints/images';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { addToast } from 'features/system/store/systemSlice';
import { useTranslation } from 'react-i18next';

const MultipleSelectionMenuItems = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selection = useAppSelector((state) => state.gallery.selection);
  const customStarUi = useStore($customStarUI);

  const isBulkDownloadEnabled =
    useFeatureStatus('bulkDownload').isFeatureEnabled;

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

  const handleBulkDownload = useCallback(async () => {
    try {
      const response = await bulkDownload({
        image_names: selection.map((img) => img.image_name),
      }).unwrap();

      dispatch(
        addToast({
          title: t('gallery.preparingDownload'),
          status: 'success',
          ...(response.response ? { description: response.response } : {}),
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('gallery.preparingDownloadFailed'),
          status: 'error',
        })
      );
    }
  }, [t, selection, bulkDownload, dispatch]);

  const areAllStarred = useMemo(() => {
    return selection.every((img) => img.starred);
  }, [selection]);

  const areAllUnstarred = useMemo(() => {
    return selection.every((img) => !img.starred);
  }, [selection]);

  return (
    <>
      {areAllStarred && (
        <MenuItem
          icon={customStarUi ? customStarUi.on.icon : <MdStarBorder />}
          onClickCapture={handleUnstarSelection}
        >
          {customStarUi ? customStarUi.off.text : `Unstar All`}
        </MenuItem>
      )}
      {(areAllUnstarred || (!areAllStarred && !areAllUnstarred)) && (
        <MenuItem
          icon={customStarUi ? customStarUi.on.icon : <MdStar />}
          onClickCapture={handleStarSelection}
        >
          {customStarUi ? customStarUi.on.text : `Star All`}
        </MenuItem>
      )}
      {isBulkDownloadEnabled && (
        <MenuItem icon={<FaDownload />} onClickCapture={handleBulkDownload}>
          {t('gallery.downloadSelection')}
        </MenuItem>
      )}
      <MenuItem icon={<FaFolder />} onClickCapture={handleChangeBoard}>
        {t('boards.changeBoard')}
      </MenuItem>
      <MenuItem
        sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
        icon={<FaTrash />}
        onClickCapture={handleDeleteSelection}
      >
        {t('gallery.deleteSelection')}
      </MenuItem>
    </>
  );
};

export default memo(MultipleSelectionMenuItems);
