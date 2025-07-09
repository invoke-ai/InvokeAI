import { useIsRegionFocused } from 'common/hooks/focus';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { useCallback, useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

export const useDeleteImage = (imageDTO?: ImageDTO | null) => {
  const deleteImageModal = useDeleteImageModalApi();
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');

  const isEnabled = useMemo(() => {
    if (!imageDTO) {
      return;
    }
    if (!isGalleryFocused && !isViewerFocused) {
      return false;
    }
    return true;
  }, [imageDTO, isGalleryFocused, isViewerFocused]);
  const _delete = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!isEnabled) {
      return;
    }
    deleteImageModal.delete([imageDTO.image_name]);
  }, [deleteImageModal, imageDTO, isEnabled]);

  return {
    delete: _delete,
    isEnabled,
  };
};
