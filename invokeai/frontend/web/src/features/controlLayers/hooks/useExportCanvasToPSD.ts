import { type Layer, type Psd, writePsd } from 'ag-psd';
import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { downloadBlob } from 'features/controlLayers/konva/util';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const log = logger('canvas');

// Canvas size limits for PSD export
// These are conservative limits to prevent memory issues with large canvases
// The actual limit may be lower depending on available memory
const MAX_CANVAS_DIMENSION = 8192; // 8K resolution
const MAX_CANVAS_AREA = MAX_CANVAS_DIMENSION * MAX_CANVAS_DIMENSION; // ~64MP max

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

      const visibleRect = canvasManager.compositor.getRectOfAdapters(adapters);

      if (visibleRect.width <= 0 || visibleRect.height <= 0) {
        toast({
          id: 'INVALID_CANVAS_DIMENSIONS',
          title: t('toast.invalidCanvasDimensions'),
          status: 'error',
        });
        return;
      }

      const canvasArea = visibleRect.width * visibleRect.height;
      if (canvasArea > MAX_CANVAS_AREA) {
        toast({
          id: 'CANVAS_TOO_LARGE',
          title: t('toast.canvasTooLarge'),
          description: t('toast.canvasTooLargeDesc'),
          status: 'error',
        });
        return;
      }

      log.debug(`PSD canvas dimensions: ${visibleRect.width}x${visibleRect.height}`);

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

      const psd: Psd = {
        width: visibleRect.width,
        height: visibleRect.height,
        channels: 3,
        bitsPerChannel: 8,
        colorMode: 3, // RGB mode
        children: psdLayers,
      };

      log.debug(
        {
          layerCount: psd.children?.length ?? 0,
          canvasDimensions: { width: psd.width, height: psd.height },
          layers:
            psd.children?.map((l) => ({
              name: l.name,
              bounds: { left: l.left, top: l.top, right: l.right, bottom: l.bottom },
            })) ?? [],
        },
        'Creating PSD with layers'
      );

      const buffer = writePsd(psd);

      const blob = new Blob([buffer], { type: 'application/octet-stream' });
      const fileName = `canvas-layers-${new Date().toISOString().slice(0, 10)}.psd`;
      downloadBlob(blob, fileName);

      toast({
        id: 'PSD_EXPORT_SUCCESS',
        title: t('toast.psdExportSuccess'),
        description: t('toast.psdExportSuccessDesc', { count: psd.children?.length ?? 0 }),
        status: 'success',
      });

      log.debug('Successfully exported canvas to PSD');
    } catch (error) {
      log.error({ error: parseify(error) }, 'Problem exporting canvas to PSD');
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
