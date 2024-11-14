import { Alert, AlertDescription, AlertIcon, AlertTitle } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { selectSystemShouldShowInvocationProgressDetail } from 'features/system/store/systemSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { $invocationProgressMessage } from 'services/events/stores';

const CanvasAlertsInvocationProgressContent = memo(() => {
  const { t } = useTranslation();
  const invocationProgressMessage = useStore($invocationProgressMessage);

  if (!invocationProgressMessage) {
    return null;
  }

  return (
    <Alert status="loading" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
      <AlertIcon />
      <AlertTitle>{t('common.generating')}</AlertTitle>
      <AlertDescription>{invocationProgressMessage}</AlertDescription>
    </Alert>
  );
});
CanvasAlertsInvocationProgressContent.displayName = 'CanvasAlertsInvocationProgressContent';

export const CanvasAlertsInvocationProgress = memo(() => {
  const isProgressMessageAlertEnabled = useFeatureStatus('invocationProgressAlert');
  const shouldShowInvocationProgressDetail = useAppSelector(selectSystemShouldShowInvocationProgressDetail);

  // The alert is disabled at the system level
  if (!isProgressMessageAlertEnabled) {
    return null;
  }

  // The alert is disabled at the user level
  if (!shouldShowInvocationProgressDetail) {
    return null;
  }

  return <CanvasAlertsInvocationProgressContent />;
});

CanvasAlertsInvocationProgress.displayName = 'CanvasAlertsInvocationProgress';
