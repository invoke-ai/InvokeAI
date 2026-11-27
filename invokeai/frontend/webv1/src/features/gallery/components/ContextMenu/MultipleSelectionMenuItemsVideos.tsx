import { MenuDivider, MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useDownloadItem } from 'common/hooks/useDownloadImage';
import { isModalOpenChanged, videosToChangeSelected } from 'features/changeBoardModal/store/slice';
import { useDeleteVideoModalApi } from 'features/deleteVideoModal/store/state';
import { selectSelection } from 'features/gallery/store/gallerySelectors';
import { isVideoName } from 'features/gallery/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold, PiFoldersBold, PiStarBold, PiStarFill, PiTrashSimpleBold } from 'react-icons/pi';
import { getVideoDTOSafe, useStarVideosMutation, useUnstarVideosMutation } from 'services/api/endpoints/videos';
import { useBoardAccess } from 'services/api/hooks/useBoardAccess';
import { useSelectedBoard } from 'services/api/hooks/useSelectedBoard';

/**
 * Multi-selection menu surfaced by `VideoContextMenu` when the gallery selection has more
 * than one item. Mirrors `MultipleSelectionMenuItems` (the image equivalent) feature-for-feature
 * where the video API supports it. Filters the polymorphic gallery selection down to videos —
 * mixed selections that also contain images are handled by the image-side menu when the user
 * right-clicks an image.
 */
const MultipleSelectionMenuItemsVideos = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selection = useAppSelector(selectSelection);
  const deleteVideoModal = useDeleteVideoModalApi();
  const selectedBoard = useSelectedBoard();
  // Boards use one write permission for both kinds — videos inherit from `canWriteImages`.
  const { canWriteImages } = useBoardAccess(selectedBoard);

  const [starVideos] = useStarVideosMutation();
  const [unstarVideos] = useUnstarVideosMutation();
  const { downloadItem } = useDownloadItem();

  const videoNames = useMemo(() => selection.filter(isVideoName), [selection]);
  const count = videoNames.length;
  const hasVideos = count > 0;

  const handleChangeBoard = useCallback(() => {
    dispatch(videosToChangeSelected(videoNames));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, videoNames]);

  const handleDeleteSelection = useCallback(() => {
    deleteVideoModal.delete(videoNames).catch(() => {
      // user cancelled the confirmation dialog
    });
  }, [deleteVideoModal, videoNames]);

  const handleStarSelection = useCallback(() => {
    starVideos({ video_names: videoNames });
  }, [starVideos, videoNames]);

  const handleUnstarSelection = useCallback(() => {
    unstarVideos({ video_names: videoNames });
  }, [unstarVideos, videoNames]);

  const handleBulkDownload = useCallback(async () => {
    // No zip-bundle endpoint exists for videos, so we loop the per-video download helper.
    // Modern browsers prompt once for "allow multiple file downloads", then proceed silently.
    for (const video_name of videoNames) {
      const dto = await getVideoDTOSafe(video_name);
      if (!dto) {
        continue;
      }
      await downloadItem(dto.video_url, dto.video_name);
    }
  }, [downloadItem, videoNames]);

  return (
    <>
      <MenuItem icon={<PiStarBold />} onClickCapture={handleUnstarSelection} isDisabled={!hasVideos}>
        {t('gallery.unstarVideo', { count })}
      </MenuItem>
      <MenuItem icon={<PiStarFill />} onClickCapture={handleStarSelection} isDisabled={!hasVideos}>
        {t('gallery.starVideo', { count })}
      </MenuItem>
      <MenuItem icon={<PiDownloadSimpleBold />} onClickCapture={handleBulkDownload} isDisabled={!hasVideos}>
        {t('gallery.downloadVideo', { count })}
      </MenuItem>
      <MenuItem icon={<PiFoldersBold />} onClickCapture={handleChangeBoard} isDisabled={!hasVideos || !canWriteImages}>
        {t('boards.changeBoardVideo', { count })}
      </MenuItem>
      <MenuDivider />
      <MenuItem
        color="error.300"
        icon={<PiTrashSimpleBold />}
        onClickCapture={handleDeleteSelection}
        isDisabled={!hasVideos || !canWriteImages}
      >
        {t('gallery.deleteVideo', { count })}
      </MenuItem>
    </>
  );
};

export default memo(MultipleSelectionMenuItemsVideos);
