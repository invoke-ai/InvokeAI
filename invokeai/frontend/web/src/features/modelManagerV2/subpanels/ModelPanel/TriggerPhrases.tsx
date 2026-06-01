import {
  Button,
  Flex,
  FormControl,
  FormErrorMessage,
  FormLabel,
  Input,
  Tag,
  TagCloseButton,
  TagLabel,
} from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import type { ChangeEvent } from 'react';
import React, { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowClockwiseBold, PiPlusBold } from 'react-icons/pi';
import { useRefreshModelTriggerPhrasesMutation, useUpdateModelMutation } from 'services/api/endpoints/models';
import type { LoRAModelConfig, MainModelConfig } from 'services/api/types';

type Props = {
  modelConfig: MainModelConfig | LoRAModelConfig;
};

export const TriggerPhrases = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const [phrase, setPhrase] = useState('');

  const [updateModel, { isLoading }] = useUpdateModelMutation();
  const [refreshModelTriggerPhrases, { isLoading: isRefreshing }] = useRefreshModelTriggerPhrasesMutation();

  const handlePhraseChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setPhrase(e.target.value);
  }, []);

  const triggerPhrases = useMemo(() => {
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
    if (!phrase.length || triggerPhrases.includes(phrase)) {
      return;
    }

    setPhrase('');

    await updateModel({
      key: modelConfig.key,
      body: { trigger_phrases: [...triggerPhrases, phrase] },
    }).unwrap();
  }, [phrase, triggerPhrases, updateModel, modelConfig.key]);

  const removeTriggerPhrase = useCallback(
    async (phraseToRemove: string) => {
      const filteredPhrases = triggerPhrases.filter((p) => p !== phraseToRemove);

      await updateModel({ key: modelConfig.key, body: { trigger_phrases: filteredPhrases } }).unwrap();
    },
    [triggerPhrases, updateModel, modelConfig]
  );

  const onTriggerPhraseAddFormSubmit = useCallback(
    (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      addTriggerPhrase();
    },
    [addTriggerPhrase]
  );

  const refreshTriggerPhrases = useCallback(() => {
    refreshModelTriggerPhrases({ key: modelConfig.key })
      .unwrap()
      .then(() => {
        toast({
          id: 'TRIGGER_PHRASES_REFRESHED',
          title: t('modelManager.triggerPhrasesRefreshed'),
          status: 'success',
        });
      })
      .catch(() => {
        toast({
          id: 'TRIGGER_PHRASES_REFRESH_FAILED',
          title: t('modelManager.triggerPhrasesRefreshFailed'),
          status: 'error',
        });
      });
  }, [modelConfig.key, refreshModelTriggerPhrases, t]);

  const hasTriggerPhrases = triggerPhrases.length > 0;
  const canRefreshTriggerPhrases = modelConfig.type === 'lora';

  return (
    <Flex flexDir="column" w="full" gap="5">
      <form onSubmit={onTriggerPhraseAddFormSubmit}>
        <FormControl w="full" isInvalid={Boolean(errors.length)} orientation="vertical">
          <FormLabel>{t('modelManager.triggerPhrases')}</FormLabel>
          <Flex flexDir="column" w="full">
            <Flex gap="3" alignItems="center" w="full">
              <Input value={phrase} onChange={handlePhraseChange} placeholder={t('modelManager.typePhraseHere')} />
              <Button
                leftIcon={<PiPlusBold />}
                size="sm"
                onClick={addTriggerPhrase}
                isDisabled={!phrase || Boolean(errors.length)}
                isLoading={isLoading}
              >
                {t('common.add')}
              </Button>
              {canRefreshTriggerPhrases && (
                <Button
                  leftIcon={<PiArrowClockwiseBold />}
                  size="sm"
                  onClick={refreshTriggerPhrases}
                  isLoading={isRefreshing}
                  isDisabled={isLoading}
                >
                  {t('modelManager.refreshTriggerPhrases')}
                </Button>
              )}
            </Flex>
            {errors.map((error) => (
              <FormErrorMessage key={error}>{error}</FormErrorMessage>
            ))}
          </Flex>
        </FormControl>
      </form>

      {hasTriggerPhrases && (
        <Flex gap="4" flexWrap="wrap">
          {triggerPhrases.map((phrase, index) => (
            <Tag size="md" key={index} py={2} px={4} bg="base.700">
              <TagLabel>{phrase}</TagLabel>
              <TagCloseButton onClick={removeTriggerPhrase.bind(null, phrase)} isDisabled={isLoading || isRefreshing} />
            </Tag>
          ))}
        </Flex>
      )}
    </Flex>
  );
});

TriggerPhrases.displayName = 'TriggerPhrases';
