import { useDeleteVideoModalApi } from 'features/deleteVideoModal/store/state';
import { useCallback, useMemo } from 'react';
import type { VideoDTO } from 'services/api/types';

export const useDeleteVideo = (videoDTO?: VideoDTO | null) => {
  const deleteImageModal = useDeleteVideoModalApi();

  const isEnabled = useMemo(() => {
    if (!videoDTO) {
      return;
    }
    return true;
  }, [videoDTO]);
  const _delete = useCallback(() => {
    if (!videoDTO) {
      return;
    }
    if (!isEnabled) {
      return;
    }
    deleteImageModal.delete([videoDTO.video_id]);
  }, [deleteImageModal, videoDTO, isEnabled]);

  return {
    delete: _delete,
    isEnabled,
  };
};
