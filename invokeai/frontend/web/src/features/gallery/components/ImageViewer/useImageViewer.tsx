import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { useCallback } from 'react';

export const TAB_NAME_TO_TKEY: Record<InvokeTabName, string> = {
  generation: 'ui.tabs.generation',
  canvas: 'ui.tabs.canvas',
  workflows: 'ui.tabs.workflows',
  models: 'ui.tabs.models',
  queue: 'ui.tabs.queue',
};

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
