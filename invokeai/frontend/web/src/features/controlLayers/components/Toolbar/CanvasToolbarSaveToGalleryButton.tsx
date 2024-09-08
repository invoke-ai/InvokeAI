import { IconButton, useShiftModifier } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { isOk, withResultAsync } from 'common/util/result';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';

const log = logger('canvas');

const [useIsSaving] = buildUseBoolean(false);

export const CanvasToolbarSaveToGalleryButton = memo(() => {
  const { t } = useTranslation();
  const shift = useShiftModifier();
  const canvasManager = useCanvasManager();
  const isSaving = useIsSaving();

  const onClick = useCallback(async () => {
    isSaving.setTrue();

    const rect = shift ? canvasManager.stateApi.getBbox().rect : canvasManager.stage.getVisibleRect('raster_layer');

    const result = await withResultAsync(() =>
      canvasManager.compositor.rasterizeAndUploadCompositeRasterLayer(rect, true)
    );

    if (isOk(result)) {
      toast({ title: t('controlLayers.savedToGalleryOk') });
    } else {
      log.error({ error: serializeError(result.error) }, 'Failed to save canvas to gallery');
      toast({ title: t('controlLayers.savedToGalleryError'), status: 'error' });
    }

    isSaving.setFalse();
  }, [canvasManager.compositor, canvasManager.stage, canvasManager.stateApi, isSaving, shift, t]);

  return (
    <IconButton
      variant="ghost"
      onClick={onClick}
      icon={<PiFloppyDiskBold />}
      isLoading={isSaving.isTrue}
      aria-label={shift ? t('controlLayers.saveBboxToGallery') : t('controlLayers.saveCanvasToGallery')}
      tooltip={shift ? t('controlLayers.saveBboxToGallery') : t('controlLayers.saveCanvasToGallery')}
    />
  );
});

CanvasToolbarSaveToGalleryButton.displayName = 'CanvasToolbarSaveToGalleryButton';
