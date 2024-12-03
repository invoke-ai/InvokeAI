import { Button, Text, useToast } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch } from 'app/store/storeHooks';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { t } from 'i18next';
import { atom } from 'nanostores';
import { useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetHFTokenStatusQuery } from 'services/api/endpoints/models';
import type { S } from 'services/api/types';

const FEATURE_ID = 'hfToken';
const TOAST_ID = 'hfTokenLogin';
/**
 * Tracks whether or not the HF Login toast is showing
 */
export const $isHFLoginToastOpen = atom<boolean>(false);

const getTitle = (token_status: S['HFTokenStatus']) => {
  switch (token_status) {
    case 'invalid':
      return t('modelManager.hfTokenInvalid');
    case 'unknown':
      return t('modelManager.hfTokenUnableToVerify');
  }
};

export const useHFLoginToast = () => {
  const isEnabled = useFeatureStatus(FEATURE_ID);
  const { data } = useGetHFTokenStatusQuery(isEnabled ? undefined : skipToken);
  const toast = useToast();
  const isHFLoginToastOpen = useStore($isHFLoginToastOpen);

  useEffect(() => {
    if (!isHFLoginToastOpen) {
      toast.close(TOAST_ID);
      return;
    }

    if (isHFLoginToastOpen && data) {
      const title = getTitle(data);
      toast({
        id: TOAST_ID,
        title,
        description: <ToastDescription token_status={data} />,
        status: 'error',
        isClosable: true,
        duration: null,
        onCloseComplete: () => $isHFLoginToastOpen.set(false),
      });
    }
  }, [isHFLoginToastOpen, data, toast]);
};

type Props = {
  token_status: S['HFTokenStatus'];
};

const ToastDescription = ({ token_status }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const toast = useToast();

  const onClick = useCallback(() => {
    dispatch(setActiveTab('models'));
    toast.close(FEATURE_ID);
  }, [dispatch, toast]);

  if (token_status === 'invalid') {
    return (
      <Text fontSize="md">
        {t('modelManager.hfTokenInvalidErrorMessage')} {t('modelManager.hfTokenRequired')}{' '}
        {t('modelManager.hfTokenInvalidErrorMessage2')}
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
