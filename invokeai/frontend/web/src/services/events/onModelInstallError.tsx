import { Button, ExternalLink, Spinner, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { logger } from 'app/logging/logger';
import type { AppDispatch, RootState } from 'app/store/store';
import { useAppDispatch } from 'app/store/storeHooks';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { toast, toastApi } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { t } from 'i18next';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { api } from 'services/api';
import { modelsApi, useGetHFTokenStatusQuery } from 'services/api/endpoints/models';
import type { S } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('events');
const selectModelInstalls = modelsApi.endpoints.listModelInstalls.select();

export const UNAUTHORIZED_TOAST_ID = getPrefixedId('unauthorized-toast');
const FORBIDDEN_TOAST_ID = getPrefixedId('forbidden-toast');

const getHFTokenStatusToastTitle = (token_status: S['HFTokenStatus']) => {
  switch (token_status) {
    case 'invalid':
      return t('modelManager.hfTokenInvalid');
    case 'unknown':
      return t('modelManager.hfTokenUnableToVerify');
    case 'valid':
      return 'HF Token is valid, but does not have access to model';
  }
};

const getHFTokenStatus = async (dispatch: AppDispatch): Promise<S['HFTokenStatus'] | null> => {
  const req = dispatch(modelsApi.endpoints.getHFTokenStatus.initiate(undefined, { subscribe: false }));
  try {
    return await req.unwrap();
  } catch {
    return null;
  }
};

export const buildOnModelInstallError = (getState: () => RootState, dispatch: AppDispatch) => {
  return async (data: S['ModelInstallErrorEvent']) => {
    log.error({ data }, 'Model install error');

    const installs = selectModelInstalls(getState()).data;

    if (data.error === 'Unauthorized') {
      if (data.source.type === 'hf') {
        const hfTokenStatus = await getHFTokenStatus(dispatch);
        if (hfTokenStatus === null) {
          // This means there was a problem getting the token status, should never happen...
          log.error('Error getting HF token status');
        } else {
          const title = getHFTokenStatusToastTitle(hfTokenStatus);
          toast({
            id: UNAUTHORIZED_TOAST_ID,
            title,
            description: <HFUnauthorizedToastDescription />,
            status: 'error',
            isClosable: true,
            duration: null,
          });
        }
      } else if (data.source.type === 'url') {
        toast({
          id: UNAUTHORIZED_TOAST_ID,
          title: 'Model requires authorization',
          description: (
            <Text fontSize="md">
              {t('modelManager.urlUnauthorizedErrorMessage')}
              <ExternalLink
                label={t('modelManager.urlUnauthorizedErrorMessage2')}
                href="https://invoke-ai.github.io/InvokeAI/configuration/#model-marketplace-api-keys"
              />
            </Text>
          ),
          status: 'error',
          isClosable: true,
          duration: null,
        });
      } else {
        // data.source.type === 'path' - should never happen
        assert(false, 'Unexpected unauthorized error for path source');
      }
    } else if (data.error === 'Forbidden') {
      if (data.source.type === 'hf') {
        toast({
          id: FORBIDDEN_TOAST_ID,
          title: t('modelManager.hfForbidden'),
          description: (
            <Text fontSize="md">
              {t('modelManager.hfForbiddenErrorMessage')}
              <ExternalLink label={data.source.repo_id} href={`https://huggingface.co/${data.source.repo_id}`} />
            </Text>
          ),
          status: 'error',
          isClosable: true,
          duration: null,
        });
      } else if (data.source.type === 'url') {
        toast({
          id: FORBIDDEN_TOAST_ID,
          title: t('modelManager.urlForbidden'),
          description: t('modelManager.urlForbiddenErrorMessage'),
          status: 'error',
          isClosable: true,
          duration: null,
        });
      } else {
        // data.source.type === 'path' - should never happen
        assert(false, 'Unexpected forbidden error for path source');
      }
    } else {
      // Some other type of error, who knows!
      toast({
        title: 'Model install error',
        description: data.error,
        status: 'error',
        isClosable: true,
        duration: null,
      });
    }

    const install = installs?.find((install) => install.id === data.id);

    if (!install) {
      dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
    } else {
      install.source;
      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === data.id);
          if (modelImport) {
            modelImport.status = 'error';
            modelImport.error_reason = data.error_type;
            modelImport.error = data.error;
          }
          return draft;
        })
      );
    }
  };
};

const HFUnauthorizedToastDescription = () => {
  const isEnabled = useFeatureStatus('hfToken');
  const { data } = useGetHFTokenStatusQuery(isEnabled ? undefined : skipToken);

  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    dispatch(setActiveTab('models'));
    toastApi.close(UNAUTHORIZED_TOAST_ID);
  }, [dispatch]);

  if (!data) {
    return <Spinner />;
  }

  if (data === 'invalid') {
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

  if (data === 'unknown') {
    return (
      <Text fontSize="md">
        {t('modelManager.hfTokenUnableToErrorMessage')}{' '}
        <Button onClick={onClick} variant="link" color="base.50" flexGrow={0}>
          {t('modelManager.modelManager')}.
        </Button>
      </Text>
    );
  }

  // data === 'valid' - should never happen!
  assert(false, 'Unexpected valid HF token with unauthorized error');
};
