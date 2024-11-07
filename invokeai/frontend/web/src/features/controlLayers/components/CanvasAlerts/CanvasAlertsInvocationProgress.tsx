import { Alert, AlertIcon, AlertTitle } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { memo } from 'react';
import { $progressEventMessage } from 'services/events/stores';

export const CanvasAlertsInvocationProgress = memo(() => {
  const progressEventMessage = useStore($progressEventMessage);

  if (!progressEventMessage) {
    return <></>;
  }

  return (
    <Alert status="loading" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
      <AlertIcon />
      <AlertTitle>{progressEventMessage}</AlertTitle>
    </Alert>
  );
});

CanvasAlertsInvocationProgress.displayName = 'CanvasAlertsInvocationProgress';
