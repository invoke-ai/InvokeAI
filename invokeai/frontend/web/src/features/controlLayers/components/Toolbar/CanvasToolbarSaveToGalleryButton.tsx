import { IconButton, useShiftModifier } from '@invoke-ai/ui-library';
import { useSaveBboxToGallery, useSaveCanvasToGallery } from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

export const CanvasToolbarSaveToGalleryButton = memo(() => {
  const { t } = useTranslation();
  const shift = useShiftModifier();
  const isBusy = useCanvasIsBusy();
  const saveCanvasToGallery = useSaveCanvasToGallery();
  const saveBboxToGallery = useSaveBboxToGallery();

  return (
    <IconButton
      variant="link"
      alignSelf="stretch"
      onClick={shift ? saveBboxToGallery : saveCanvasToGallery}
      icon={<PiFloppyDiskBold />}
      aria-label={shift ? t('controlLayers.saveBboxToGallery') : t('controlLayers.saveCanvasToGallery')}
      tooltip={shift ? t('controlLayers.saveBboxToGallery') : t('controlLayers.saveCanvasToGallery')}
      isDisabled={isBusy}
    />
  );
});

CanvasToolbarSaveToGalleryButton.displayName = 'CanvasToolbarSaveToGalleryButton';
