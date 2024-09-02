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
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type { LoRAModelConfig, MainModelConfig } from 'services/api/types';

type Props = {
  modelConfig: MainModelConfig | LoRAModelConfig;
};

export const TriggerPhrases = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const [phrase, setPhrase] = useState('');

  const [updateModel, { isLoading }] = useUpdateModelMutation();

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
            </Flex>
            {errors.map((error) => (
              <FormErrorMessage key={error}>{error}</FormErrorMessage>
            ))}
          </Flex>
        </FormControl>
      </form>

      <Flex gap="4" flexWrap="wrap">
        {triggerPhrases.map((phrase, index) => (
          <Tag size="md" key={index} py={2} px={4} bg="base.700">
            <TagLabel>{phrase}</TagLabel>
            <TagCloseButton onClick={removeTriggerPhrase.bind(null, phrase)} isDisabled={isLoading} />
          </Tag>
        ))}
      </Flex>
    </Flex>
  );
});

TriggerPhrases.displayName = 'TriggerPhrases';
