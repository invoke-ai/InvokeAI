import { useStore } from '@nanostores/react';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { $invocationProgressMessage } from 'services/events/stores';

export const useDeferredModelLoadingInvocationProgressMessage = () => {
  const { t } = useTranslation();
  const invocationProgressMessage = useStore($invocationProgressMessage);
  const [delayedMessage, setDelayedMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!invocationProgressMessage) {
      setDelayedMessage(null);
      return;
    }

    if (invocationProgressMessage && !invocationProgressMessage.startsWith('Loading model')) {
      setDelayedMessage(null);
      return;
    }

    // Set a timeout to update delayedMessage after 5 seconds
    const timer = setTimeout(() => {
      setDelayedMessage(`${t('common.loadingModel')}...`);
    }, 5000);

    return () => clearTimeout(timer); // Cleanup on effect re-run
  }, [invocationProgressMessage, t]);

  return delayedMessage;
};
