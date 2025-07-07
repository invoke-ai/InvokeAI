import { Alert, AlertIcon, AlertTitle } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSaveAllImagesToGallery } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasAlertsSaveAllImagesToGallery = memo(() => {
  const { t } = useTranslation();
  const saveAllImagesToGallery = useAppSelector(selectSaveAllImagesToGallery);

  if (!saveAllImagesToGallery) {
    return null;
  }

  return (
    <Alert status="info" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
      <AlertIcon />
      <AlertTitle>{t('controlLayers.settings.saveAllImagesToGallery.alert')}</AlertTitle>
    </Alert>
  );
});

CanvasAlertsSaveAllImagesToGallery.displayName = 'CanvasAlertsSaveAllImagesToGallery';
