import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imageToCompareChanged, isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';

export const useIsImageViewerOpen = () => {
  const isOpen = useAppSelector((s) => {
    const tab = s.ui.activeTab;
    const workflowsMode = s.workflow.mode;
    if (tab === 'models' || tab === 'queue') {
      return false;
    }
    if (tab === 'workflows' && workflowsMode === 'edit') {
      return false;
    }
    if (tab === 'workflows' && workflowsMode === 'view') {
      return true;
    }
    if (tab === 'upscaling') {
      return true;
    }
    return s.gallery.isImageViewerOpen;
  });
  return isOpen;
};

export const useImageViewer = () => {
  const dispatch = useAppDispatch();
  const isComparing = useAppSelector((s) => s.gallery.imageToCompare !== null);
  const isNaturallyOpen = useAppSelector((s) => s.gallery.isImageViewerOpen);
  const isForcedOpen = useAppSelector(
    (s) => s.ui.activeTab === 'upscaling' || (s.ui.activeTab === 'workflows' && s.workflow.mode === 'view')
  );

  const onClose = useCallback(() => {
    if (isForcedOpen) {
      return;
    }
    if (isComparing && isNaturallyOpen) {
      dispatch(imageToCompareChanged(null));
    } else {
      dispatch(isImageViewerOpenChanged(false));
    }
  }, [dispatch, isComparing, isForcedOpen, isNaturallyOpen]);

  const onOpen = useCallback(() => {
    dispatch(isImageViewerOpenChanged(true));
  }, [dispatch]);

  const onToggle = useCallback(() => {
    if (isForcedOpen) {
      return;
    }
    if (isComparing && isNaturallyOpen) {
      dispatch(imageToCompareChanged(null));
    } else {
      dispatch(isImageViewerOpenChanged(!isNaturallyOpen));
    }
  }, [dispatch, isComparing, isForcedOpen, isNaturallyOpen]);

  return { isOpen: isNaturallyOpen || isForcedOpen, onOpen, onClose, onToggle, isComparing };
};
