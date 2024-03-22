import { Button, Text, useToast } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch } from 'app/store/storeHooks';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { t } from 'i18next';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetHFTokenStatusQuery } from 'services/api/endpoints/models';
import type { S } from 'services/api/types';

const FEATURE_ID = 'hfToken';

const getTitle = (token_status: S['HFTokenStatus']) => {
  switch (token_status) {
    case 'invalid':
      return t('modelManager.hfTokenInvalid');
    case 'unknown':
      return t('modelManager.hfTokenUnableToVerify');
  }
};

export const useHFLoginToast = () => {
  const { t } = useTranslation();
  const isEnabled = useFeatureStatus(FEATURE_ID).isFeatureEnabled;
  const [didToast, setDidToast] = useState(false);
  const { data } = useGetHFTokenStatusQuery(isEnabled ? undefined : skipToken);
  const toast = useToast();

  useEffect(() => {
    if (toast.isActive(FEATURE_ID)) {
      if (data === 'valid') {
        setDidToast(true);
        toast.close(FEATURE_ID);
      }
      return;
    }
    if (data && data !== 'valid' && !didToast && isEnabled) {
      const title = getTitle(data);
      toast({
        id: FEATURE_ID,
        title,
        description: <ToastDescription token_status={data} />,
        status: 'info',
        isClosable: true,
        duration: null,
        onCloseComplete: () => setDidToast(true),
      });
    }
  }, [data, didToast, isEnabled, t, toast]);
};

type Props = {
  token_status: S['HFTokenStatus'];
};

const ToastDescription = ({ token_status }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const toast = useToast();

  const onClick = useCallback(() => {
    dispatch(setActiveTab('modelManager'));
    toast.close(FEATURE_ID);
  }, [dispatch, toast]);

  if (token_status === 'invalid') {
    return (
      <Text fontSize="md">
        {t('modelManager.hfTokenInvalidErrorMessage')} {t('modelManager.hfTokenInvalidErrorMessage2')}
        <Button onClick={onClick} variant="link" color="base.50" flexGrow={0}>
          {t('modelManager.modelManager')}.
        </Button>
      </Text>
    );
  }

  if (token_status === 'unknown') {
    return (
      <Text fontSize="md">
        {t('modelManager.hfTokenUnableToErrorMessage')}{' '}
        <Button onClick={onClick} variant="link" color="base.50" flexGrow={0}>
          {t('modelManager.modelManager')}.
        </Button>
      </Text>
    );
  }
};
