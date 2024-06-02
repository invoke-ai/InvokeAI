import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imageToCompareChanged, isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';

export const useImageViewer = () => {
  const dispatch = useAppDispatch();
  const isComparing = useAppSelector((s) => s.gallery.imageToCompare !== null);
  const isOpen = useAppSelector((s) => s.gallery.isImageViewerOpen);

  const onClose = useCallback(() => {
    if (isComparing && isOpen) {
      dispatch(imageToCompareChanged(null));
    } else {
      dispatch(isImageViewerOpenChanged(false));
    }
  }, [dispatch, isComparing, isOpen]);

  const onOpen = useCallback(() => {
    dispatch(isImageViewerOpenChanged(true));
  }, [dispatch]);

  const onToggle = useCallback(() => {
    if (isComparing && isOpen) {
      dispatch(imageToCompareChanged(null));
    } else {
      dispatch(isImageViewerOpenChanged(!isOpen));
    }
  }, [dispatch, isComparing, isOpen]);

  return { isOpen, onOpen, onClose, onToggle };
};
