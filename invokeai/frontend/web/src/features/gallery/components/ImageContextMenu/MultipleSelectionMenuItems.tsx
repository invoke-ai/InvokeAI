import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { InvMenuDivider } from 'common/components/InvMenu/wrapper'
import {
  imagesToChangeSelected,
  isModalOpenChanged,
} from 'features/changeBoardModal/store/slice';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold, PiFoldersBold, PiStarBold, PiStarFill, PiTrashSimpleBold } from 'react-icons/pi'
import {
  useBulkDownloadImagesMutation,
  useStarImagesMutation,
  useUnstarImagesMutation,
} from 'services/api/endpoints/images';

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
          ...(response.response
            ? {
                description: response.response,
                duration: null,
                isClosable: true,
              }
            : {}),
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
        <InvMenuItem
          icon={customStarUi ? customStarUi.on.icon : <PiStarBold />}
          onClickCapture={handleUnstarSelection}
        >
          {customStarUi ? customStarUi.off.text : `Unstar All`}
        </InvMenuItem>
      )}
      {(areAllUnstarred || (!areAllStarred && !areAllUnstarred)) && (
        <InvMenuItem
          icon={customStarUi ? customStarUi.on.icon : <PiStarFill />}
          onClickCapture={handleStarSelection}
        >
          {customStarUi ? customStarUi.on.text : `Star All`}
        </InvMenuItem>
      )}
      {isBulkDownloadEnabled && (
        <InvMenuItem icon={<PiDownloadSimpleBold />} onClickCapture={handleBulkDownload}>
          {t('gallery.downloadSelection')}
        </InvMenuItem>
      )}
      <InvMenuItem icon={<PiFoldersBold />} onClickCapture={handleChangeBoard}>
        {t('boards.changeBoard')}
      </InvMenuItem>
      <InvMenuDivider />
      <InvMenuItem
        color="error.300"
        icon={<PiTrashSimpleBold />}
        onClickCapture={handleDeleteSelection}
      >
        {t('gallery.deleteSelection')}
      </InvMenuItem>
    </>
  );
};

export default memo(MultipleSelectionMenuItems);
