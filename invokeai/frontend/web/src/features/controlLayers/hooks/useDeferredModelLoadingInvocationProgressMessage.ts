import { useStore } from '@nanostores/react';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { $lastProgressMessage } from 'services/events/stores';

export const useDeferredModelLoadingInvocationProgressMessage = () => {
  const { t } = useTranslation();
  const invocationProgressMessage = useStore($lastProgressMessage);
  const [delayedInvocationProgressMessage, setDelayedInvocationProgressMessage] = useState<string | null>(null);
  const isLoadingModelProgressMessage = invocationProgressMessage?.startsWith('Loading model') ?? false;

  useEffect(() => {
    if (!isLoadingModelProgressMessage) {
      return;
    }

    const timer = setTimeout(() => {
      setDelayedInvocationProgressMessage(invocationProgressMessage);
    }, 5000);

    return () => clearTimeout(timer);
  }, [invocationProgressMessage, isLoadingModelProgressMessage]);

  if (!isLoadingModelProgressMessage || delayedInvocationProgressMessage !== invocationProgressMessage) {
    return null;
  }

  return `${t('common.loadingModel')}...`;
};
