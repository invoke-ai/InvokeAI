import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { viewerModeChanged } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';

export const useImageViewer = () => {
  const dispatch = useAppDispatch();
  const viewerMode = useAppSelector((s) => s.gallery.viewerMode);

  const openEditor = useCallback(() => {
    dispatch(viewerModeChanged('edit'));
  }, [dispatch]);

  const openViewer = useCallback(() => {
    dispatch(viewerModeChanged('view'));
  }, [dispatch]);

  const onToggle = useCallback(() => {
    dispatch(viewerModeChanged(viewerMode === 'view' ? 'edit' : 'view'));
  }, [dispatch, viewerMode]);

  const openCompare = useCallback(() => {
    dispatch(viewerModeChanged('compare'));
  }, [dispatch]);

  return { viewerMode, openEditor, openViewer, openCompare, onToggle };
};
