import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { canvasToBlob } from 'features/controlLayers/konva/util';
import { selectRasterLayerEntities } from 'features/controlLayers/store/selectors';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { serializeError } from 'serialize-error';
import { writePsd } from 'ag-psd';

const log = logger('canvas');

export const useExportCanvasToPSD = () => {
  const { t } = useTranslation();
  const rasterLayers = useAppSelector(selectRasterLayerEntities);
  const canvasManager = useCanvasManagerSafe();

  const exportCanvasToPSD = useCallback(async () => {
    try {
      if (!canvasManager) {
        toast({
          id: 'CANVAS_MANAGER_NOT_AVAILABLE',
          title: t('toast.canvasManagerNotAvailable'),
          status: 'error',
        });
        return;
      }

      if (rasterLayers.length === 0) {
        toast({
          id: 'NO_RASTER_LAYERS',
          title: t('toast.noRasterLayers'),
          description: t('toast.noRasterLayersDesc'),
          status: 'warning',
        });
        return;
      }

      // Get active (enabled) raster layers only
      const activeLayers = rasterLayers.filter((layer) => layer.isEnabled);

      if (activeLayers.length === 0) {
        toast({
          id: 'NO_ACTIVE_RASTER_LAYERS',
          title: t('toast.noActiveRasterLayers'),
          description: t('toast.noActiveRasterLayersDesc'),
          status: 'warning',
        });
        return;
      }

      log.debug(`Exporting ${activeLayers.length} active raster layers to PSD`);

      // Find canvas dimensions by getting the maximum bounds of all layers
      let maxRight = 0;
      let maxBottom = 0;
      let minLeft = Infinity;
      let minTop = Infinity;

      // Get layer adapters and calculate bounds
      const layerAdapters = activeLayers.map((layer) => {
        const adapter = canvasManager.getAdapter({ id: layer.id, type: 'raster_layer' });
        if (adapter && adapter.type === 'raster_layer_adapter') {
          const canvas = adapter.getCanvas();
          const rect = adapter.transformer.$pixelRect.get();
          
          const left = layer.position.x + rect.x;
          const top = layer.position.y + rect.y;
          const right = left + rect.width;
          const bottom = top + rect.height;

          minLeft = Math.min(minLeft, left);
          minTop = Math.min(minTop, top);
          maxRight = Math.max(maxRight, right);
          maxBottom = Math.max(maxBottom, bottom);

          return { adapter, canvas, rect: { x: left, y: top, width: rect.width, height: rect.height }, layer };
        }
        return null;
      }).filter((item): item is NonNullable<typeof item> => item !== null);

      if (layerAdapters.length === 0) {
        toast({
          id: 'NO_VALID_LAYER_ADAPTERS',
          title: t('toast.noValidLayerAdapters'),
          status: 'error',
        });
        return;
      }

      // Default canvas size if no valid bounds found
      const canvasWidth = maxRight > minLeft ? Math.ceil(maxRight - minLeft) : 1024;
      const canvasHeight = maxBottom > minTop ? Math.ceil(maxBottom - minTop) : 1024;

      log.debug(`PSD canvas dimensions: ${canvasWidth}x${canvasHeight}`);

      // Create PSD layers from active raster layers
      const psdLayers = await Promise.all(
        layerAdapters.map(async (layerData, index) => {
          try {
            const { adapter, canvas, rect, layer } = layerData;
            
            const layerDataPSD = {
              name: layer.name || `Layer ${index + 1}`,
              left: Math.floor(rect.x - minLeft),
              top: Math.floor(rect.y - minTop),
              right: Math.floor(rect.x - minLeft + rect.width),
              bottom: Math.floor(rect.y - minTop + rect.height),
              opacity: Math.floor(layer.opacity * 255),
              hidden: false,
              blendMode: 'normal' as const,
              canvas: canvas,
            };

            log.debug(`Layer "${layerDataPSD.name}": ${layerDataPSD.left},${layerDataPSD.top} to ${layerDataPSD.right},${layerDataPSD.bottom}`);

            return layerDataPSD;
          } catch (error) {
            log.error({ error: serializeError(error as Error) }, `Error processing layer ${layerData.layer.name}`);
            return null;
          }
        })
      );

      // Filter out any failed layers
      const validLayers = psdLayers.filter((layer) => layer !== null);

      if (validLayers.length === 0) {
        toast({
          id: 'FAILED_TO_PROCESS_LAYERS',
          title: t('toast.failedToProcessLayers'),
          status: 'error',
        });
        return;
      }

      // Create PSD document
      const psd = {
        width: canvasWidth,
        height: canvasHeight,
        channels: 3, // RGB
        bitsPerChannel: 8,
        colorMode: 3, // RGB mode
        children: validLayers,
      };

      log.debug({ layerCount: validLayers.length }, 'Creating PSD with layers');

      // Generate PSD file
      const buffer = writePsd(psd);
      
      // Create blob and download
      const blob = new Blob([buffer], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `canvas-layers-${new Date().toISOString().slice(0, 10)}.psd`;
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast({
        id: 'PSD_EXPORT_SUCCESS',
        title: t('toast.psdExportSuccess'),
        description: t('toast.psdExportSuccessDesc', { count: validLayers.length }),
        status: 'success',
      });

      log.debug('Successfully exported canvas to PSD');
    } catch (error) {
      log.error({ error: serializeError(error as Error) }, 'Problem exporting canvas to PSD');
      toast({
        id: 'PROBLEM_EXPORTING_PSD',
        title: t('toast.problemExportingPSD'),
        description: String(error),
        status: 'error',
      });
    }
  }, [rasterLayers, canvasManager, t]);

  return { exportCanvasToPSD };
};