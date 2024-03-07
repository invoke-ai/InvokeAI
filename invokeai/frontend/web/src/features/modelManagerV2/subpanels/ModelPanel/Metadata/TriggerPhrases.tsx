import {
  Button,
  Flex,
  FormControl,
  FormErrorMessage,
  Input,
  Tag,
  TagCloseButton,
  TagLabel,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { ModelListHeader } from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelListHeader';
import type { ChangeEvent } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetModelConfigQuery, useUpdateModelMutation } from 'services/api/endpoints/models';
import { isNonRefinerMainModelConfig } from 'services/api/types';

export const TriggerPhrases = () => {
  const { t } = useTranslation();
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data: modelConfig } = useGetModelConfigQuery(selectedModelKey ?? skipToken);
  const [phrase, setPhrase] = useState('');

  const [updateModel, { isLoading }] = useUpdateModelMutation();

  const handlePhraseChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setPhrase(e.target.value);
  }, []);

  const triggerPhrases = useMemo(() => {
    if (!modelConfig || !isNonRefinerMainModelConfig(modelConfig)) {
      return [];
    }
    return modelConfig?.trigger_phrases || [];
  }, [modelConfig]);

  const errors = useMemo(() => {
    const errors = [];

    if (phrase.length && triggerPhrases.includes(phrase)) {
      errors.push('Phrase is already in list');
    }

    return errors;
  }, [phrase, triggerPhrases]);

  const addTriggerPhrase = useCallback(async () => {
    if (!selectedModelKey) {
      return;
    }

    if (!phrase.length || triggerPhrases.includes(phrase)) {
      return;
    }

    await updateModel({
      key: selectedModelKey,
      body: { trigger_phrases: [...triggerPhrases, phrase] },
    }).unwrap();
    setPhrase('');
  }, [updateModel, selectedModelKey, phrase, triggerPhrases]);

  const removeTriggerPhrase = useCallback(
    async (phraseToRemove: string) => {
      if (!selectedModelKey) {
        return;
      }

      const filteredPhrases = triggerPhrases.filter((p) => p !== phraseToRemove);

      await updateModel({ key: selectedModelKey, body: { trigger_phrases: filteredPhrases } }).unwrap();
    },
    [updateModel, selectedModelKey, triggerPhrases]
  );

  return (
    <Flex flexDir="column" w="full" gap="5">
      <ModelListHeader title={t('modelManager.triggerPhrases')} />
      <form>
        <FormControl w="full" isInvalid={Boolean(errors.length)}>
          <Flex flexDir="column" w="full">
            <Flex gap="3" alignItems="center" w="full">
              <Input value={phrase} onChange={handlePhraseChange} placeholder={t('modelManager.typePhraseHere')} />
              <Button
                type="submit"
                onClick={addTriggerPhrase}
                isDisabled={Boolean(errors.length)}
                isLoading={isLoading}
              >
                {t('common.add')}
              </Button>
            </Flex>
            {!!errors.length && errors.map((error) => <FormErrorMessage key={error}>{error}</FormErrorMessage>)}
          </Flex>
        </FormControl>
      </form>

      <Flex gap="4" flexWrap="wrap" mt="3" mb="3">
        {triggerPhrases.map((phrase, index) => (
          <Tag size="md" key={index}>
            <TagLabel>{phrase}</TagLabel>
            <TagCloseButton onClick={removeTriggerPhrase.bind(null, phrase)} isDisabled={isLoading} />
          </Tag>
        ))}
      </Flex>
    </Flex>
  );
};
