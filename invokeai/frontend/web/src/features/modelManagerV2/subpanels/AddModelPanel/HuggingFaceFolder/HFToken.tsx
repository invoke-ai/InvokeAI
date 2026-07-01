import {
  Button,
  ExternalLink,
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  Input,
} from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useGetHFTokenStatusQuery,
  useResetHFTokenMutation,
  useSetHFTokenMutation,
} from 'services/api/endpoints/models';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

export const HFToken = () => {
  const { t } = useTranslation();
  const { currentData } = useGetHFTokenStatusQuery();

  const error = useMemo(() => {
    switch (currentData) {
      case 'invalid':
        return t('modelManager.hfTokenInvalidErrorMessage');
      case 'unknown':
        return t('modelManager.hfTokenUnableToVerifyErrorMessage');
      case 'valid':
      case undefined:
        return null;
      default:
        assert<Equals<never, typeof currentData>>(false, 'Unexpected HF token status');
    }
  }, [currentData, t]);

  if (!currentData) {
    return null;
  }

  return (
    <Flex borderRadius="base" w="full">
      <FormControl isInvalid={Boolean(error)} orientation="vertical">
        <FormLabel>{t('modelManager.hfTokenLabel')}</FormLabel>
        {error && <SetHFTokenInput />}
        {!error && <ResetHFTokenButton />}
        <FormHelperText>
          <ExternalLink label={t('modelManager.hfTokenHelperText')} href="https://huggingface.co/settings/tokens" />
        </FormHelperText>
        <FormErrorMessage>{error}</FormErrorMessage>
      </FormControl>
    </Flex>
  );
};

const PLACEHOLDER_TOKEN = Array.from({ length: 37 }, () => 'a').join('');

const ResetHFTokenButton = memo(() => {
  const { t } = useTranslation();
  const [resetHFToken, { isLoading }] = useResetHFTokenMutation();

  const onClick = useCallback(() => {
    resetHFToken()
      .unwrap()
      .then(() => {
        toast({
          title: t('modelManager.hfTokenReset'),
          status: 'info',
        });
      });
  }, [resetHFToken, t]);

  return (
    <Flex gap={3} alignItems="center" w="full">
      <Input type="password" value={PLACEHOLDER_TOKEN} isDisabled />
      <Button onClick={onClick} size="sm" isLoading={isLoading}>
        {t('common.reset')}
      </Button>
    </Flex>
  );
});
ResetHFTokenButton.displayName = 'ResetHFTokenButton';

const SetHFTokenInput = memo(() => {
  const { t } = useTranslation();
  const [token, setToken] = useState('');
  const [trigger, { isLoading }] = useSetHFTokenMutation();
  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setToken(e.target.value);
  }, []);
  const onClick = useCallback(() => {
    trigger({ token })
      .unwrap()
      .then((res) => {
        switch (res) {
          case 'valid':
            setToken('');
            toast({
              title: t('modelManager.hfTokenSaved'),
              status: 'success',
            });
            break;
          case 'invalid':
          case 'unknown':
          default:
            toast({
              title: t('modelManager.hfTokenUnableToVerify'),
              status: 'error',
            });
            break;
        }
      });
  }, [t, token, trigger]);

  return (
    <Flex gap={3} alignItems="center" w="full">
      <Input type="password" value={token} onChange={onChange} />
      <Button onClick={onClick} size="sm" isDisabled={token.trim().length === 0} isLoading={isLoading}>
        {t('common.save')}
      </Button>
    </Flex>
  );
});
SetHFTokenInput.displayName = 'SetHFTokenInput';
