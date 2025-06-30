import { writePsd, type Layer, type Psd } from 'ag-psd';
import { logger } from 'app/logging/logger';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { downloadBlob } from 'features/controlLayers/konva/util';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { serializeError } from 'serialize-error';

const log = logger('canvas');

export const useExportCanvasToPSD = () => {
  const { t } = useTranslation();
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

      // Get visible raster layer adapters using the compositor module
      const adapters = canvasManager.compositor.getVisibleAdaptersOfType('raster_layer');

      if (adapters.length === 0) {
        toast({
          id: 'NO_VISIBLE_RASTER_LAYERS',
          title: t('toast.noVisibleRasterLayers'),
          description: t('toast.noVisibleRasterLayersDesc'),
          status: 'warning',
        });
        return;
      }

      log.debug(`Exporting ${adapters.length} visible raster layers to PSD`);

      // Get the union rect of all adapters using the compositor module
      const visibleRect = canvasManager.compositor.getRectOfAdapters(adapters);

      // Validate canvas dimensions
      if (visibleRect.width <= 0 || visibleRect.height <= 0) {
        toast({
          id: 'INVALID_CANVAS_DIMENSIONS',
          title: t('toast.invalidCanvasDimensions'),
          status: 'error',
        });
        return;
      }

      if (visibleRect.width > 30000 || visibleRect.height > 30000) {
        toast({
          id: 'CANVAS_TOO_LARGE',
          title: t('toast.canvasTooLarge'),
          description: t('toast.canvasTooLargeDesc'),
          status: 'error',
        });
        return;
      }

      log.debug(`PSD canvas dimensions: ${visibleRect.width}x${visibleRect.height}`);

      // Create PSD layers from visible raster layer adapters
      const psdLayers: Layer[] = await Promise.all(
        adapters.map((adapter, index) => {
          const layer = adapter.state;
          const canvas = adapter.getCanvas();
          const layerPosition = adapter.state.position;

          const layerDataPSD: Layer = {
            name: layer.name || `Layer ${index + 1}`,
            left: Math.floor(layerPosition.x - visibleRect.x),
            top: Math.floor(layerPosition.y - visibleRect.y),
            right: Math.floor(layerPosition.x - visibleRect.x + canvas.width),
            bottom: Math.floor(layerPosition.y - visibleRect.y + canvas.height),
            opacity: Math.floor(layer.opacity * 255),
            hidden: false,
            blendMode: 'normal',
            canvas: canvas,
          };

          log.debug(
            `Layer "${layerDataPSD.name}": ${layerDataPSD.left},${layerDataPSD.top} to ${layerDataPSD.right},${layerDataPSD.bottom}`
          );

          return layerDataPSD;
        })
      );

      // Create PSD document
      const psd: Psd = {
        width: visibleRect.width,
        height: visibleRect.height,
        channels: 3, // RGB
        bitsPerChannel: 8,
        colorMode: 3, // RGB mode
        children: psdLayers,
      };

      log.debug(
        {
          layerCount: psdLayers.length,
          canvasDimensions: { width: psd.width, height: psd.height },
          layers: psdLayers.map((l) => ({
            name: l.name,
            bounds: { left: l.left, top: l.top, right: l.right, bottom: l.bottom },
          })),
        },
        'Creating PSD with layers'
      );

      // Generate PSD file
      const buffer = writePsd(psd);

      // Create blob and download using the utility function
      const blob = new Blob([buffer], { type: 'application/octet-stream' });
      const fileName = `canvas-layers-${new Date().toISOString().slice(0, 10)}.psd`;
      downloadBlob(blob, fileName);

      toast({
        id: 'PSD_EXPORT_SUCCESS',
        title: t('toast.psdExportSuccess'),
        description: t('toast.psdExportSuccessDesc', { count: psdLayers.length }),
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
  }, [canvasManager, t]);

  return { exportCanvasToPSD };
};
