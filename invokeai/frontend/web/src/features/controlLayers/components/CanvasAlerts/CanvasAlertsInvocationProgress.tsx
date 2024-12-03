import { Alert, AlertDescription, AlertIcon, AlertTitle } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useDeferredModelLoadingInvocationProgressMessage } from 'features/controlLayers/hooks/useDeferredModelLoadingInvocationProgressMessage';
import { selectIsLocal } from 'features/system/store/configSlice';
import { selectSystemShouldShowInvocationProgressDetail } from 'features/system/store/systemSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { $invocationProgressMessage } from 'services/events/stores';

const CanvasAlertsInvocationProgressContentLocal = memo(() => {
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
CanvasAlertsInvocationProgressContentLocal.displayName = 'CanvasAlertsInvocationProgressContentLocal';

const CanvasAlertsInvocationProgressContentCommercial = memo(() => {
  const message = useDeferredModelLoadingInvocationProgressMessage();

  if (!message) {
    return null;
  }

  return (
    <Alert status="loading" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
      <AlertIcon />
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  );
});
CanvasAlertsInvocationProgressContentCommercial.displayName = 'CanvasAlertsInvocationProgressContentCommercial';

export const CanvasAlertsInvocationProgress = memo(() => {
  const shouldShowInvocationProgressDetail = useAppSelector(selectSystemShouldShowInvocationProgressDetail);
  const isLocal = useAppSelector(selectIsLocal);

  if (!isLocal) {
    return <CanvasAlertsInvocationProgressContentCommercial />;
  }

  // OSS user setting
  if (!shouldShowInvocationProgressDetail) {
    return null;
  }

  return <CanvasAlertsInvocationProgressContentLocal />;
});

CanvasAlertsInvocationProgress.displayName = 'CanvasAlertsInvocationProgress';
