import { Alert, AlertIcon, AlertTitle } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSaveAllStagingImagesToGallery } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasAlertsSaveAllStagingImagesToGallery = memo(() => {
  const { t } = useTranslation();
  const saveAllStagingImagesToGallery = useAppSelector(selectSaveAllStagingImagesToGallery);

  if (!saveAllStagingImagesToGallery) {
    return null;
  }

  return (
    <Alert status="info" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
      <AlertIcon />
      <AlertTitle>{t('controlLayers.settings.saveAllStagingImagesToGallery.alert')}</AlertTitle>
    </Alert>
  );
});

CanvasAlertsSaveAllStagingImagesToGallery.displayName = 'CanvasAlertsSaveAllStagingImagesToGallery';
