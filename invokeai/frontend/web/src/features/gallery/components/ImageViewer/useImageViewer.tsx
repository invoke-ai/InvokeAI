import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';

export const useImageViewer = () => {
  const dispatch = useAppDispatch();
  const isOpen = useAppSelector((s) => s.gallery.isImageViewerOpen);

  const onClose = useCallback(() => {
    dispatch(isImageViewerOpenChanged(false));
  }, [dispatch]);

  const onOpen = useCallback(() => {
    dispatch(isImageViewerOpenChanged(true));
  }, [dispatch]);

  const onToggle = useCallback(() => {
    dispatch(isImageViewerOpenChanged(!isOpen));
  }, [dispatch, isOpen]);

  return { isOpen, onOpen, onClose, onToggle };
};
