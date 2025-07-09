import { useAppStore } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { ImageDTO } from 'services/api/types';

export const useEditImage = (imageDTO?: ImageDTO | null) => {
  const { t } = useTranslation();
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');

  const { getState, dispatch } = useAppStore();
  const canvasManager = useCanvasManagerSafe();

  const isEnabled = useMemo(() => {
    if (!imageDTO) {
      return false;
    }
    if (!isGalleryFocused && !isViewerFocused) {
      return false;
    }
    return true;
  }, [imageDTO, isGalleryFocused, isViewerFocused]);

  const edit = useCallback(async () => {
    if (!imageDTO) {
      return;
    }

    if (!isEnabled) {
      return;
    }

    await newCanvasFromImage({
      imageDTO,
      type: 'raster_layer',
      withInpaintMask: true,
      getState,
      dispatch,
    });
    navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);

    if (canvasManager) {
      canvasManager.tool.$tool.set('brush');
    }

    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, isEnabled, getState, dispatch, canvasManager, t]);

  return {
    edit,
    isEnabled,
  };
};
