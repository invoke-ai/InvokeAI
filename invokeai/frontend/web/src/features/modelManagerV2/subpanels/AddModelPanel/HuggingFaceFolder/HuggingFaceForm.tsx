import { Button, Flex, FormControl, FormErrorMessage, FormHelperText, FormLabel, Input } from '@invoke-ai/ui-library';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetHuggingFaceModelsQuery } from 'services/api/endpoints/models';

import { HuggingFaceResults } from './HuggingFaceResults';

export const HuggingFaceForm = memo(() => {
  const [huggingFaceRepo, setHuggingFaceRepo] = useState('');
  const [displayResults, setDisplayResults] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const { t } = useTranslation();

  const [_getHuggingFaceModels, { isLoading, data }] = useLazyGetHuggingFaceModelsQuery();
  const [installModel] = useInstallModel();

  const getModels = useCallback(() => {
    _getHuggingFaceModels(huggingFaceRepo)
      .unwrap()
      .then((response) => {
        if (response.is_diffusers) {
          installModel({ source: huggingFaceRepo });
          setDisplayResults(false);
        } else if (response.urls?.length === 1 && response.urls[0]) {
          installModel({ source: response.urls[0] });
          setDisplayResults(false);
        } else {
          setDisplayResults(true);
        }
      })
      .catch((error) => {
        setErrorMessage(error.data.detail || '');
      });
  }, [_getHuggingFaceModels, installModel, huggingFaceRepo]);

  const handleSetHuggingFaceRepo: ChangeEventHandler<HTMLInputElement> = useCallback((e) => {
    setHuggingFaceRepo(e.target.value);
    setErrorMessage('');
  }, []);

  return (
    <Flex flexDir="column" height="100%" gap={3}>
      <FormControl isInvalid={!!errorMessage.length} w="full" orientation="vertical" flexShrink={0}>
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
            {t('modelManager.installRepo')}
          </Button>
        </Flex>
        <FormHelperText>{t('modelManager.huggingFaceHelper')}</FormHelperText>
        {!!errorMessage.length && <FormErrorMessage>{errorMessage}</FormErrorMessage>}
      </FormControl>
      {data && data.urls && displayResults && <HuggingFaceResults results={data.urls} />}
    </Flex>
  );
});

HuggingFaceForm.displayName = 'HuggingFaceForm';
