import { IconButton, useShiftModifier } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
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
    <IAITooltip label={shift ? t('controlLayers.saveBboxToGallery') : t('controlLayers.saveCanvasToGallery')}>
      <IconButton
        variant="link"
        alignSelf="stretch"
        onClick={shift ? saveBboxToGallery : saveCanvasToGallery}
        icon={<PiFloppyDiskBold />}
        aria-label={shift ? t('controlLayers.saveBboxToGallery') : t('controlLayers.saveCanvasToGallery')}
        colorScheme="invokeBlue"
        isDisabled={isBusy}
      />
    </IAITooltip>
  );
});

CanvasToolbarSaveToGalleryButton.displayName = 'CanvasToolbarSaveToGalleryButton';
