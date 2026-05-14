import { useAppSelector } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useDeleteVideoModalApi } from 'features/deleteVideoModal/store/state';
import { useVideoDTOContext } from 'features/gallery/contexts/VideoDTOContext';
import { selectSelection } from 'features/gallery/store/gallerySelectors';
import { isVideoName } from 'features/gallery/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const ContextMenuItemDeleteVideo = memo(() => {
  const { t } = useTranslation();
  const videoDTO = useVideoDTOContext();
  const deleteVideoModal = useDeleteVideoModalApi();
  const selection = useAppSelector(selectSelection);

  // When the right-clicked video is part of an active multi-selection, delete every
  // selected video in one shot. Image names mixed into the selection are skipped —
  // right-clicking an image surfaces ImageContextMenu, which owns that flow.
  const targetVideoNames = useMemo(() => {
    if (selection.length > 1 && selection.includes(videoDTO.video_name)) {
      return selection.filter(isVideoName);
    }
    return [videoDTO.video_name];
  }, [selection, videoDTO.video_name]);

  const label = t('gallery.deleteVideo', { count: targetVideoNames.length });

  const onClick = useCallback(async () => {
    try {
      await deleteVideoModal.delete(targetVideoNames);
    } catch {
      // noop — user canceled the confirm dialog.
    }
  }, [deleteVideoModal, targetVideoNames]);

  return (
    <IconMenuItem
      icon={<PiTrashSimpleBold />}
      onClickCapture={onClick}
      aria-label={label}
      tooltip={label}
      isDestructive
    />
  );
});

ContextMenuItemDeleteVideo.displayName = 'ContextMenuItemDeleteVideo';
