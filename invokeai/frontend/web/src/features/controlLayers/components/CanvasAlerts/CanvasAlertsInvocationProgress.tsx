import { Alert, AlertDescription, AlertIcon, AlertTitle } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSystemShouldShowInvocationProgressDetail } from 'features/system/store/systemSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { $canvasProgressMessage } from 'services/events/stores';

export const CanvasAlertsInvocationProgress = memo(() => {
  const { t } = useTranslation();
  const progressEventMessage = useStore($canvasProgressMessage);
  const shouldShowInvocationProgressDetail = useAppSelector(selectSystemShouldShowInvocationProgressDetail);

  if (!shouldShowInvocationProgressDetail || !progressEventMessage) {
    return null;
  }

  return (
    <Alert status="loading" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
      <AlertIcon />
      <AlertTitle>{t('common.generating')}</AlertTitle>
      <AlertDescription>{progressEventMessage}</AlertDescription>
    </Alert>
  );
});

CanvasAlertsInvocationProgress.displayName = 'CanvasAlertsInvocationProgress';
