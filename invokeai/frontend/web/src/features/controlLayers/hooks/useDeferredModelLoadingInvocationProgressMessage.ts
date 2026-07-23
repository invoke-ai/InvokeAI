import { useStore } from '@nanostores/react';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { $lastProgressMessage } from 'services/events/stores';

export const useDeferredModelLoadingInvocationProgressMessage = () => {
  const { t } = useTranslation();
  const invocationProgressMessage = useStore($lastProgressMessage);
  const [didDelayLoadingModelProgressMessage, setDidDelayLoadingModelProgressMessage] = useState(false);
  const isLoadingModelProgressMessage = invocationProgressMessage?.startsWith('Loading model') ?? false;

  useEffect(() => {
    if (!isLoadingModelProgressMessage) {
      return;
    }

    const timer = setTimeout(() => {
      setDidDelayLoadingModelProgressMessage(true);
    }, 5000);

    return () => {
      clearTimeout(timer);
      setDidDelayLoadingModelProgressMessage(false);
    };
  }, [isLoadingModelProgressMessage]);

  if (!isLoadingModelProgressMessage || !didDelayLoadingModelProgressMessage) {
    return null;
  }

  return `${t('common.loadingModel')}...`;
};
