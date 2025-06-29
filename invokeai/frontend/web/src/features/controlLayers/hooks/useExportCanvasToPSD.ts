import { writePsd } from 'ag-psd';
import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectRasterLayerEntities } from 'features/controlLayers/store/selectors';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { serializeError } from 'serialize-error';

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

      // Ensure all rect calculations are up to date before proceeding
      const rectCalculationPromises = activeLayers.map(async (layer) => {
        const adapter = canvasManager.getAdapter({ id: layer.id, type: 'raster_layer' });
        if (adapter && adapter.type === 'raster_layer_adapter') {
          await adapter.transformer.requestRectCalculation();
        }
      });

      // Wait for all rect calculations to complete with a timeout
      try {
        await Promise.race([
          Promise.all(rectCalculationPromises),
          new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Rect calculation timeout')), 5000);
          }),
        ]);
      } catch (error) {
        log.warn({ error: serializeError(error as Error) }, 'Rect calculation timeout or error, proceeding anyway');
      }

      // Find canvas dimensions by getting the maximum bounds of all layers
      let maxRight = 0;
      let maxBottom = 0;
      let minLeft = Infinity;
      let minTop = Infinity;

      // Get layer adapters and calculate bounds
      const layerAdapters = activeLayers
        .map((layer) => {
          const adapter = canvasManager.getAdapter({ id: layer.id, type: 'raster_layer' });
          log.debug(`Layer "${layer.name}": adapter found = ${!!adapter}, type = ${adapter?.type}`);

          if (adapter && adapter.type === 'raster_layer_adapter') {
            // Get the actual pixel bounds of the layer content from the transformer
            const pixelRect = adapter.transformer.$pixelRect.get();
            const layerPosition = adapter.state.position;

            log.debug(
              `Layer "${layer.name}": pixelRect=${JSON.stringify(pixelRect)}, position=${JSON.stringify(layerPosition)}`
            );

            // Alternative approach: use full canvas and adjust positioning
            const canvas = adapter.getCanvas();
            const left = layerPosition.x;
            const top = layerPosition.y;
            const right = left + canvas.width;
            const bottom = top + canvas.height;

            log.debug(
              `Layer "${layer.name}": canvas size = ${canvas.width}x${canvas.height}, position = ${left},${top}`
            );

            // Skip layers with invalid canvas dimensions
            if (canvas.width === 0 || canvas.height === 0) {
              log.debug(`Layer "${layer.name}": skipping due to invalid canvas dimensions`);
              return null;
            }

            minLeft = Math.min(minLeft, left);
            minTop = Math.min(minTop, top);
            maxRight = Math.max(maxRight, right);
            maxBottom = Math.max(maxBottom, bottom);

            // Temporarily remove the empty bounds filter to see what's happening
            // if (pixelRect.width === 0 || pixelRect.height === 0) {
            //   log.debug(`Layer "${layer.name}": skipping due to empty bounds`);
            //   return null;
            // }

            return { adapter, canvas, rect: { x: left, y: top, width: canvas.width, height: canvas.height }, layer };
          }
          return null;
        })
        .filter((item): item is NonNullable<typeof item> => item !== null);

      log.debug(`Found ${layerAdapters.length} valid layer adapters out of ${activeLayers.length} active layers`);

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

      // Validate canvas dimensions
      if (canvasWidth <= 0 || canvasHeight <= 0) {
        toast({
          id: 'INVALID_CANVAS_DIMENSIONS',
          title: t('toast.invalidCanvasDimensions'),
          status: 'error',
        });
        return;
      }

      if (canvasWidth > 30000 || canvasHeight > 30000) {
        toast({
          id: 'CANVAS_TOO_LARGE',
          title: t('toast.canvasTooLarge'),
          description: t('toast.canvasTooLargeDesc'),
          status: 'error',
        });
        return;
      }

      log.debug(`PSD canvas dimensions: ${canvasWidth}x${canvasHeight}`);

      // Create PSD layers from active raster layers
      const psdLayers = await Promise.all(
        layerAdapters.map((layerData, index) => {
          try {
            const { adapter: _adapter, canvas, rect, layer } = layerData;

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

            log.debug(
              `Layer "${layerDataPSD.name}": ${layerDataPSD.left},${layerDataPSD.top} to ${layerDataPSD.right},${layerDataPSD.bottom}`
            );

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

      log.debug(
        {
          layerCount: validLayers.length,
          canvasDimensions: { width: canvasWidth, height: canvasHeight },
          layers: validLayers.map((l) => ({
            name: l.name,
            bounds: { left: l.left, top: l.top, right: l.right, bottom: l.bottom },
          })),
        },
        'Creating PSD with layers'
      );

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
