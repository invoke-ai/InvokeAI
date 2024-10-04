import { useAppDispatch } from 'app/store/storeHooks';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { imageSelected, imageToCompareChanged } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';
import type { ImageDTO } from 'services/api/types';

/**
 * There's a race condition that causes the canvas to not fit to layers on the very first app startup.
 *
 * The canvas stage uses a resize observer to fit the stage to the container, and on the first resize event, it also
 * fits the layers to the stage. Subsequent resize events only fit the stage to the container, they do not fit layers
 * to the stage.
 *
 * On the very first app startup (new user or after they reset all web UI state), the resizable panels library needs
 * to do one extra resize as it initializes and figures out its target size. At this time, the canvas stage has already
 * done its one-time fit layers to stage, so the canvas stage does not fit layers to the stage again.
 *
 * For the end user, this means that the bbox is not centered in the canvas stage on the very first app startup. On
 * all subsequent app startups, the bbox is centered in the canvas stage.
 *
 * We can hack around this, thanks to the fact that the image viewer is always opened on the first app startup. By the
 * time the user closes it, the resizable panels library has already done its one extra resize and the DOM layout has
 * stablized. So we can track the first time the image viewer is closed and fit the layers to the stage at that time,
 * ensuring that the bbox is centered in the canvas stage on that first app startup.
 *
 * TODO(psyche): Figure out a better way to do handle this...
 */
let didCloseImageViewer = false;
const api = buildUseBoolean(true);
const useImageViewerState = api[0];
export const $imageViewer = api[1];

export const useImageViewer = () => {
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManagerSafe();
  const imageViewerState = useImageViewerState();
  const close = useCallback(() => {
    if (!didCloseImageViewer && canvasManager) {
      didCloseImageViewer = true;
      canvasManager.stage.fitLayersToStage();
    }
    imageViewerState.setFalse();
  }, [canvasManager, imageViewerState]);
  const openImageInViewer = useCallback(
    (imageDTO: ImageDTO) => {
      dispatch(imageToCompareChanged(null));
      dispatch(imageSelected(imageDTO));
      imageViewerState.setTrue();
    },
    [dispatch, imageViewerState]
  );

  return {
    isOpen: imageViewerState.isTrue,
    open: imageViewerState.setTrue,
    close,
    toggle: imageViewerState.toggle,
    $state: $imageViewer,
    openImageInViewer,
  };
};
