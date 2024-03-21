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
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import type { ChangeEvent } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetHFTokenStatusQuery, useSetHFTokenMutation } from 'services/api/endpoints/models';

export const HFToken = () => {
  const { t } = useTranslation();
  const isEnabled = useFeatureStatus('hfToken').isFeatureEnabled;
  const [token, setToken] = useState('');
  const { currentData } = useGetHFTokenStatusQuery(isEnabled ? undefined : skipToken);
  const [trigger, { isLoading }] = useSetHFTokenMutation();
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
        }
      });
  }, [t, toast, token, trigger]);

  const error = useMemo(() => {
    if (!currentData || isLoading) {
      return null;
    }
    if (currentData === 'invalid') {
      return t('modelManager.hfTokenInvalidErrorMessage');
    }
    if (currentData === 'unknown') {
      return t('modelManager.hfTokenUnableToVerifyErrorMessage');
    }
    return null;
  }, [currentData, isLoading, t]);

  if (!currentData || currentData === 'valid') {
    return null;
  }

  return (
    <Flex borderRadius="base" w="full">
      <FormControl isInvalid={Boolean(error)} orientation="vertical">
        <FormLabel>{t('modelManager.hfToken')}</FormLabel>
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
