import { useAppStore } from 'app/store/storeHooks';
import { useCanvasManagerSafe } from 'features/controlLayers/hooks/useCanvasManager';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { ImageDTO } from 'services/api/types';

export const useEditImage = (imageDTO?: ImageDTO | null) => {
  const { t } = useTranslation();

  const { getState, dispatch } = useAppStore();
  const canvasManager = useCanvasManagerSafe();
  const isStaging = useCanvasIsStaging();

  const isEnabled = useMemo(() => {
    if (!imageDTO) {
      return false;
    }
    if (isStaging) {
      return false;
    }
    return true;
  }, [imageDTO, isStaging]);

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
