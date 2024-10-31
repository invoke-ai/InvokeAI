import {
  Button,
  ExternalLink,
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  Input,
  useToast,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { $isHFLoginToastOpen } from 'features/modelManagerV2/hooks/useHFLoginToast';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import type { ChangeEvent } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetHFTokenStatusQuery, useSetHFTokenMutation } from 'services/api/endpoints/models';

export const HFToken = () => {
  const { t } = useTranslation();
  const isHFTokenEnabled = useFeatureStatus('hfToken');
  const [token, setToken] = useState('');
  const { currentData } = useGetHFTokenStatusQuery(isHFTokenEnabled ? undefined : skipToken);
  const [trigger, { isLoading, isUninitialized }] = useSetHFTokenMutation();
  const toast = useToast();
  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setToken(e.target.value);
  }, []);
  const onClick = useCallback(() => {
    trigger({ token })
      .unwrap()
      .then((res) => {
        if (res === 'valid') {
          setToken('');
          toast({
            title: t('modelManager.hfTokenSaved'),
            status: 'success',
            duration: 3000,
          });
          $isHFLoginToastOpen.set(false);
        }
      });
  }, [t, toast, token, trigger]);

  const error = useMemo(() => {
    if (!currentData || isUninitialized || isLoading) {
      return null;
    }
    if (currentData === 'invalid') {
      return t('modelManager.hfTokenInvalidErrorMessage');
    }
    if (currentData === 'unknown') {
      return t('modelManager.hfTokenUnableToVerifyErrorMessage');
    }
    return null;
  }, [currentData, isLoading, isUninitialized, t]);

  if (!currentData || currentData === 'valid') {
    return null;
  }

  return (
    <Flex borderRadius="base" w="full">
      <FormControl isInvalid={!isUninitialized && Boolean(error)} orientation="vertical">
        <FormLabel>{t('modelManager.hfTokenLabel')}</FormLabel>
        <Flex gap={3} alignItems="center" w="full">
          <Input type="password" value={token} onChange={onChange} />
          <Button onClick={onClick} size="sm" isDisabled={token.trim().length === 0} isLoading={isLoading}>
            {t('common.save')}
          </Button>
        </Flex>
        <FormHelperText>
          <ExternalLink label={t('modelManager.hfTokenHelperText')} href="https://huggingface.co/settings/tokens" />
        </FormHelperText>
        <FormErrorMessage>{error}</FormErrorMessage>
      </FormControl>
    </Flex>
  );
};
