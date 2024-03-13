import { Button, Flex, FormControl, FormErrorMessage, FormLabel, Input } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { ChangeEventHandler } from 'react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useInstallModelMutation, useLazyGetHuggingFaceModelsQuery } from 'services/api/endpoints/models';

import { HuggingFaceResults } from './HuggingFaceResults';

export const HuggingFaceForm = () => {
  const [huggingFaceRepo, setHuggingFaceRepo] = useState('');
  const [displayResults, setDisplayResults] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const [_getHuggingFaceModels, { isLoading, data }] = useLazyGetHuggingFaceModelsQuery();
  const [installModel] = useInstallModelMutation();

  const handleInstallModel = useCallback(
    (source: string) => {
      installModel({ source })
        .unwrap()
        .then((_) => {
          dispatch(
            addToast(
              makeToast({
                title: t('toast.modelAddedSimple'),
                status: 'success',
              })
            )
          );
        })
        .catch((error) => {
          if (error) {
            dispatch(
              addToast(
                makeToast({
                  title: `${error.data.detail} `,
                  status: 'error',
                })
              )
            );
          }
        });
    },
    [installModel, dispatch, t]
  );

  const getModels = useCallback(async () => {
    _getHuggingFaceModels(huggingFaceRepo)
      .unwrap()
      .then((response) => {
        if (response.is_diffusers) {
          handleInstallModel(huggingFaceRepo);
          setDisplayResults(false);
        } else if (response.urls?.length === 1 && response.urls[0]) {
          handleInstallModel(response.urls[0]);
          setDisplayResults(false);
        } else {
          setDisplayResults(true);
        }
      })
      .catch((error) => {
        setErrorMessage(error.data.detail || '');
      });
  }, [_getHuggingFaceModels, handleInstallModel, huggingFaceRepo]);

  const handleSetHuggingFaceRepo: ChangeEventHandler<HTMLInputElement> = useCallback((e) => {
    setHuggingFaceRepo(e.target.value);
    setErrorMessage('');
  }, []);

  return (
    <Flex flexDir="column" height="100%">
      <FormControl isInvalid={!!errorMessage.length} w="full" orientation="vertical">
        <FormLabel>{t('modelManager.huggingFaceRepoID')}</FormLabel>
        <Flex gap={3} alignItems="center" w="full">
          <Input
            placeholder={t('modelManager.huggingFacePlaceholder')}
            value={huggingFaceRepo}
            onChange={handleSetHuggingFaceRepo}
          />
          <Button
            onClick={getModels}
            isLoading={isLoading}
            isDisabled={huggingFaceRepo.length === 0}
            size="sm"
            flexShrink={0}
          >
            {t('modelManager.addModel')}
          </Button>
        </Flex>
        {!!errorMessage.length && <FormErrorMessage>{errorMessage}</FormErrorMessage>}
      </FormControl>
      {data && data.urls && displayResults && <HuggingFaceResults results={data.urls} />}
    </Flex>
  );
};
