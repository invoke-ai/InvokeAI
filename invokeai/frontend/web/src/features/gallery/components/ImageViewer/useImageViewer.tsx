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

export const DROP_SHADOW = 'drop-shadow(0px 0px 4px rgb(0, 0, 0)) drop-shadow(0px 0px 4px rgba(0, 0, 0, 0.3))';
