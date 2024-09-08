import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectHasImageToCompare, selectIsImageViewerOpen } from 'features/gallery/store/gallerySelectors';
import {
  imageToCompareChanged,
  isImageViewerOpenChanged,
  selectGallerySlice,
} from 'features/gallery/store/gallerySlice';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { selectUiSlice } from 'features/ui/store/uiSlice';
import { useCallback } from 'react';

const selectIsOpen = createSelector(selectUiSlice, selectWorkflowSlice, selectGallerySlice, (ui, workflow, gallery) => {
  const tab = ui.activeTab;
  const workflowsMode = workflow.mode;
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
  return gallery.isImageViewerOpen;
});

export const useIsImageViewerOpen = () => {
  const isOpen = useAppSelector(selectIsOpen);
  return isOpen;
};

const selectIsForcedOpen = createSelector(selectUiSlice, selectWorkflowSlice, (ui, workflow) => {
  return ui.activeTab === 'upscaling' || (ui.activeTab === 'workflows' && workflow.mode === 'view');
});

export const useImageViewer = () => {
  const dispatch = useAppDispatch();
  const isComparing = useAppSelector(selectHasImageToCompare);
  const isNaturallyOpen = useAppSelector(selectIsImageViewerOpen);
  const isForcedOpen = useAppSelector(selectIsForcedOpen);

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
