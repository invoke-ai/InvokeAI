import { Button, Flex, FormControl, FormErrorMessage, FormLabel, Input } from '@invoke-ai/ui-library';
import type { ChangeEventHandler } from 'react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetHuggingFaceModelsQuery } from 'services/api/endpoints/models';

import { HuggingFaceResults } from './HuggingFaceResults';

export const HuggingFaceForm = () => {
  const [huggingFaceRepo, setHuggingFaceRepo] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const { t } = useTranslation();

  const [_getHuggingFaceModels, { isLoading, data }] = useLazyGetHuggingFaceModelsQuery();

  const scanFolder = useCallback(async () => {
    _getHuggingFaceModels(huggingFaceRepo).catch((error) => {
      if (error) {
        setErrorMessage(error.data.detail);
      }
    });
  }, [_getHuggingFaceModels, huggingFaceRepo]);

  const handleSetHuggingFaceRepo: ChangeEventHandler<HTMLInputElement> = useCallback((e) => {
    setHuggingFaceRepo(e.target.value);
    setErrorMessage('');
  }, []);

  return (
    <Flex flexDir="column" height="100%">
      <FormControl isInvalid={!!errorMessage.length} w="full">
        <Flex flexDir="column" w="full">
          <Flex gap={2} alignItems="flex-end" justifyContent="space-between">
            <Flex direction="column" w="full">
              <FormLabel>{t('modelManager.huggingFaceRepoID')}</FormLabel>
              <Input value={huggingFaceRepo} onChange={handleSetHuggingFaceRepo} />
            </Flex>

            <Button onClick={scanFolder} isLoading={isLoading} isDisabled={huggingFaceRepo.length === 0}>
              {t('modelManager.addModel')}
            </Button>
          </Flex>
          {!!errorMessage.length && <FormErrorMessage>{errorMessage}</FormErrorMessage>}
        </Flex>
      </FormControl>
      {data && <HuggingFaceResults results={data} />}
    </Flex>
  );
};
