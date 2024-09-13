import { IconButton, useShiftModifier } from '@invoke-ai/ui-library';
import {
  useIsSavingCanvas,
  useSaveBboxToGallery,
  useSaveCanvasToGallery,
} from 'features/controlLayers/hooks/saveCanvasHooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

export const CanvasToolbarSaveToGalleryButton = memo(() => {
  const { t } = useTranslation();
  const shift = useShiftModifier();
  const isSaving = useIsSavingCanvas();
  const saveCanvasToGallery = useSaveCanvasToGallery();
  const saveBboxToGallery = useSaveBboxToGallery();

  return (
    <IconButton
      variant="ghost"
      onClick={shift ? saveBboxToGallery : saveCanvasToGallery}
      icon={<PiFloppyDiskBold />}
      isLoading={isSaving.isTrue}
      aria-label={shift ? t('controlLayers.saveBboxToGallery') : t('controlLayers.saveCanvasToGallery')}
      tooltip={shift ? t('controlLayers.saveBboxToGallery') : t('controlLayers.saveCanvasToGallery')}
    />
  );
});

CanvasToolbarSaveToGalleryButton.displayName = 'CanvasToolbarSaveToGalleryButton';
