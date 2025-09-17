import { MenuDivider, MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppSelector } from 'app/store/storeHooks';
import { useChangeBoardModalApi } from 'features/changeBoardModal/store/state';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFoldersBold, PiStarBold, PiStarFill, PiTrashSimpleBold } from 'react-icons/pi';
import { useDeleteVideosMutation, useStarVideosMutation, useUnstarVideosMutation } from 'services/api/endpoints/videos';

const MultipleSelectionMenuItems = () => {
  const { t } = useTranslation();
  const selection = useAppSelector((s) => s.gallery.selection);
  const customStarUi = useStore($customStarUI);
  const changeBoardModal = useChangeBoardModalApi();

  const [starVideos] = useStarVideosMutation();
  const [unstarVideos] = useUnstarVideosMutation();
  const [deleteVideos] = useDeleteVideosMutation();

  const handleChangeBoard = useCallback(() => {
    changeBoardModal.openWithVideos(selection.map((s) => s.id));
  }, [changeBoardModal, selection]);

  const handleDeleteSelection = useCallback(() => {
    // TODO: Add confirm on delete and video usage functionality
    deleteVideos({ video_ids: selection.map((s) => s.id) });
  }, [deleteVideos, selection]);

  const handleStarSelection = useCallback(() => {
    starVideos({ video_ids: selection.map((s) => s.id) });
  }, [starVideos, selection]);

  const handleUnstarSelection = useCallback(() => {
    unstarVideos({ video_ids: selection.map((s) => s.id) });
  }, [unstarVideos, selection]);

  return (
    <>
      <MenuItem icon={customStarUi ? customStarUi.on.icon : <PiStarBold />} onClickCapture={handleUnstarSelection}>
        {customStarUi ? customStarUi.off.text : `Unstar All`}
      </MenuItem>
      <MenuItem icon={customStarUi ? customStarUi.on.icon : <PiStarFill />} onClickCapture={handleStarSelection}>
        {customStarUi ? customStarUi.on.text : `Star All`}
      </MenuItem>
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
