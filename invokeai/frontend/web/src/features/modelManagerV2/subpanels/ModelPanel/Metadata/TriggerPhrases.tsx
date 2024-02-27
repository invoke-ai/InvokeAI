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
import { useState, useMemo, useCallback } from 'react';
import type { ChangeEvent } from 'react';
import { ModelListHeader } from '../../ModelManagerPanel/ModelListHeader';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from '../../../../../app/store/storeHooks';
import { useGetModelMetadataQuery, useUpdateModelMetadataMutation } from '../../../../../services/api/endpoints/models';
import { useTranslation } from 'react-i18next';

export const TriggerPhrases = () => {
  const { t } = useTranslation();
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data: metadata } = useGetModelMetadataQuery(selectedModelKey ?? skipToken);
  const [phrase, setPhrase] = useState('');

  const [editModelMetadata, { isLoading }] = useUpdateModelMetadataMutation();

  const handlePhraseChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setPhrase(e.target.value);
  }, []);

  const triggerPhrases = useMemo(() => {
    return metadata?.trigger_phrases || [];
  }, [metadata?.trigger_phrases]);

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

    await editModelMetadata({
      key: selectedModelKey,
      body: { trigger_phrases: [...triggerPhrases, phrase] },
    }).unwrap();
    setPhrase('');
  }, [editModelMetadata, selectedModelKey, phrase, triggerPhrases]);

  const removeTriggerPhrase = useCallback(
    async (phraseToRemove: string) => {
      if (!selectedModelKey) {
        return;
      }

      const filteredPhrases = triggerPhrases.filter((p) => p !== phraseToRemove);

      await editModelMetadata({ key: selectedModelKey, body: { trigger_phrases: filteredPhrases } }).unwrap();
    },
    [editModelMetadata, selectedModelKey, triggerPhrases]
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
