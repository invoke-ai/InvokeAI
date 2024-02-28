import type { ChakraProps, ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import type { PromptTriggerSelectProps } from 'features/prompt/types';
import { t } from 'i18next';
import { map } from 'lodash-es';
import { useMemo } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetModelMetadataQuery, useGetTextualInversionModelsQuery } from 'services/api/endpoints/models';

const noOptionsMessage = () => t('prompt.noMatchingTriggers');

export const PromptTriggerSelect = memo(({ onSelect, onClose }: PromptTriggerSelectProps) => {
  const { t } = useTranslation();

  const currentBaseModel = useAppSelector((s) => s.generation.model?.base);
  const currentModelKey = useAppSelector((s) => s.generation.model?.key);

  const { data, isLoading } = useGetTextualInversionModelsQuery();
  const { data: metadata } = useGetModelMetadataQuery(currentModelKey ?? skipToken);

  const _onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        onSelect('');
        return;
      }

      onSelect(v.value);
    },
    [onSelect]
  );

  const embeddingOptions = useMemo(() => {
    if (!data) {
      return [];
    }

    const compatibleEmbeddingsArray = map(data.entities).filter((model) => model.base === currentBaseModel);

    return [
      {
        label: t('prompt.compatibleEmbeddings'),
        options: compatibleEmbeddingsArray.map((model) => ({ label: model.name, value: `<${model.name}>` })),
      },
    ];
  }, [data, currentBaseModel]);

  const options = useMemo(() => {
    if (!metadata || !metadata.trigger_phrases) {
      return [...embeddingOptions];
    }

    const metadataOptions = [
      {
        label: t('modelManager.triggerPhrases'),
        options: metadata.trigger_phrases.map((phrase) => ({ label: phrase, value: phrase })),
      },
    ];
    return [...metadataOptions, ...embeddingOptions];
  }, [embeddingOptions, metadata]);

  return (
    <FormControl>
      <Combobox
        placeholder={isLoading ? t('common.loading') : t('prompt.addPromptTrigger')}
        defaultMenuIsOpen
        autoFocus
        value={null}
        options={options}
        noOptionsMessage={noOptionsMessage}
        onChange={_onChange}
        onMenuClose={onClose}
        data-testid="add-prompt-trigger"
        sx={selectStyles}
      />
    </FormControl>
  );
});

PromptTriggerSelect.displayName = 'PromptTriggerSelect';

const selectStyles: ChakraProps['sx'] = {
  w: 'full',
};
