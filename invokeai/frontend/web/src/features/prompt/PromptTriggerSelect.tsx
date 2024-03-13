import type { ChakraProps, ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import type { GroupBase } from 'chakra-react-select';
import { selectLoraSlice } from 'features/lora/store/loraSlice';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import type { PromptTriggerSelectProps } from 'features/prompt/types';
import { t } from 'i18next';
import { flatten, map } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  loraModelsAdapterSelectors,
  textualInversionModelsAdapterSelectors,
  useGetLoRAModelsQuery,
  useGetModelConfigQuery,
  useGetTextualInversionModelsQuery,
} from 'services/api/endpoints/models';
import { isNonRefinerMainModelConfig } from 'services/api/types';

const noOptionsMessage = () => t('prompt.noMatchingTriggers');

const selectLoRAs = createMemoizedSelector(selectLoraSlice, (loras) => loras.loras);
const selectMainModel = createMemoizedSelector(selectGenerationSlice, (generation) => generation.model);

export const PromptTriggerSelect = memo(({ onSelect, onClose }: PromptTriggerSelectProps) => {
  const { t } = useTranslation();

  const mainModel = useAppSelector(selectMainModel);
  const addedLoRAs = useAppSelector(selectLoRAs);
  const { data: mainModelConfig, isLoading: isLoadingMainModelConfig } = useGetModelConfigQuery(
    mainModel?.key ?? skipToken
  );
  const { data: loraModels, isLoading: isLoadingLoRAs } = useGetLoRAModelsQuery();
  const { data: tiModels, isLoading: isLoadingTIs } = useGetTextualInversionModelsQuery();

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

    if (tiModels) {
      const embeddingOptions = textualInversionModelsAdapterSelectors
        .selectAll(tiModels)
        .filter((ti) => ti.base === mainModelConfig?.base)
        .map((model) => ({ label: model.name, value: `<${model.name}>` }));

      if (embeddingOptions.length > 0) {
        _options.push({
          label: t('prompt.compatibleEmbeddings'),
          options: embeddingOptions,
        });
      }
    }

    if (loraModels) {
      const triggerPhraseOptions = loraModelsAdapterSelectors
        .selectAll(loraModels)
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
  }, [tiModels, loraModels, mainModelConfig, t, addedLoRAs]);

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
      />
    </FormControl>
  );
});

PromptTriggerSelect.displayName = 'PromptTriggerSelect';

const selectStyles: ChakraProps['sx'] = {
  w: 'full',
};
