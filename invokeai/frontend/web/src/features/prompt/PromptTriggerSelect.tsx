import type { ChakraProps, ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, Icon } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import type { GroupBase } from 'chakra-react-select';
import { flatten, map } from 'es-toolkit/compat';
import { selectAddedLoRAs } from 'features/controlLayers/store/lorasSlice';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import type { PromptTriggerSelectProps } from 'features/prompt/types';
import { t } from 'i18next';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLinkSimple } from 'react-icons/pi';
import { useGetRelatedModelIdsBatchQuery } from 'services/api/endpoints/modelRelationships';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';
import { useEmbeddingModels, useLoRAModels } from 'services/api/hooks/modelsByType';
import { isNonRefinerMainModelConfig } from 'services/api/types';

const noOptionsMessage = () => t('prompt.noMatchingTriggers');

type RelatedEmbedding = ComboboxOption & { starred?: boolean };

export const PromptTriggerSelect = memo(({ onSelect, onClose }: PromptTriggerSelectProps) => {
  const { t } = useTranslation();

  const mainModel = useAppSelector(selectModel);
  const addedLoRAs = useAppSelector(selectAddedLoRAs);
  const { data: mainModelConfig, isLoading: isLoadingMainModelConfig } = useGetModelConfigQuery(
    mainModel?.key ?? skipToken
  );
  const [loraModels, { isLoading: isLoadingLoRAs }] = useLoRAModels();
  const [tiModels, { isLoading: isLoadingTIs }] = useEmbeddingModels();

  // Get related model keys for current selected models
  const selectedModelKeys = useMemo(() => {
    const keys: string[] = [];
    if (mainModel) {
      keys.push(mainModel.key);
    }
    for (const { model } of addedLoRAs) {
      keys.push(model.key);
    }
    return keys;
  }, [mainModel, addedLoRAs]);

  const { relatedModelKeys } = useGetRelatedModelIdsBatchQuery(selectedModelKeys, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { relatedModelKeys: [] };
      }
      return { relatedModelKeys: data };
    },
  });

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

  const options = useMemo(() => {
    const _options: GroupBase<ComboboxOption>[] = [];

    if (loraModels) {
      const triggerPhraseOptions = loraModels
        .filter((lora) => map(addedLoRAs, (l) => l.model.key).includes(lora.key))
        .map((lora) => {
          if (lora.trigger_phrases) {
            return lora.trigger_phrases.map((triggerPhrase) => ({ label: triggerPhrase, value: triggerPhrase }));
          }
          return [];
        })
        .flatMap((x) => x);

      if (triggerPhraseOptions.length > 0) {
        _options.push({
          label: t('modelManager.loraTriggerPhrases'),
          options: flatten(triggerPhraseOptions),
        });
      }
    }

    if (tiModels) {
      // Create embedding options with starred property for related models
      const embeddingOptions: RelatedEmbedding[] = tiModels
        .filter((ti) => ti.base === mainModelConfig?.base)
        .map((model) => ({
          label: model.name,
          value: `<${model.name}>`,
          starred: relatedModelKeys.includes(model.key),
        }));

      // Sort so related embeddings come first
      embeddingOptions.sort((a, b) => {
        if (a.starred && !b.starred) {
          return -1;
        }
        if (!a.starred && b.starred) {
          return 1;
        }
        return 0;
      });

      if (embeddingOptions.length > 0) {
        _options.push({
          label: t('prompt.compatibleEmbeddings'),
          options: embeddingOptions,
        });
      }
    }

    if (mainModelConfig && isNonRefinerMainModelConfig(mainModelConfig) && mainModelConfig.trigger_phrases?.length) {
      _options.push({
        label: t('modelManager.mainModelTriggerPhrases'),
        options: mainModelConfig.trigger_phrases.map((triggerPhrase) => ({
          label: triggerPhrase,
          value: triggerPhrase,
        })),
      });
    }

    return _options;
  }, [tiModels, loraModels, mainModelConfig, t, addedLoRAs, relatedModelKeys]);

  const formatOptionLabel = useCallback((option: ComboboxOption) => {
    const embeddingOption = option as RelatedEmbedding;
    if (embeddingOption.starred) {
      return (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Icon as={PiLinkSimple} color="invokeYellow.500" boxSize={3} />
          {option.label}
        </div>
      );
    }
    return option.label;
  }, []);

  return (
    <FormControl>
      <Combobox
        placeholder={
          isLoadingLoRAs || isLoadingTIs || isLoadingMainModelConfig
            ? t('common.loading')
            : t('prompt.addPromptTrigger')
        }
        defaultMenuIsOpen
        autoFocus
        value={null}
        options={options}
        noOptionsMessage={noOptionsMessage}
        onChange={_onChange}
        onMenuClose={onClose}
        data-testid="add-prompt-trigger"
        sx={selectStyles}
        formatOptionLabel={formatOptionLabel}
      />
    </FormControl>
  );
});

PromptTriggerSelect.displayName = 'PromptTriggerSelect';

const selectStyles: ChakraProps['sx'] = {
  w: 'full',
};
