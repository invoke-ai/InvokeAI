import type { ChakraProps, ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { GroupBase, OptionsOrGroups } from 'chakra-react-select';
import type { PromptTriggerSelectProps } from 'features/prompt/types';
import { t } from 'i18next';
import { map } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetTextualInversionModelsQuery } from 'services/api/endpoints/models';

const noOptionsMessage = () => t('prompt.noMatchingTriggers');

export const PromptTriggerSelect = memo(({ onSelect, onClose }: PromptTriggerSelectProps) => {
  const { t } = useTranslation();

  const currentBaseModel = useAppSelector((s) => s.generation.model?.base);
  const triggerPhrases = useAppSelector((s) => s.generation.triggerPhrases);

  const { data, isLoading } = useGetTextualInversionModelsQuery();

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
    let embeddingOptions: OptionsOrGroups<ComboboxOption, GroupBase<ComboboxOption>> = [];

    if (data) {
      const compatibleEmbeddingsArray = map(data.entities).filter((model) => model.base === currentBaseModel);

      embeddingOptions = [
        {
          label: t('prompt.compatibleEmbeddings'),
          options: compatibleEmbeddingsArray.map((model) => ({ label: model.name, value: `<${model.name}>` })),
        },
      ];
    }

    const metadataOptions = [
      {
        label: t('modelManager.triggerPhrases'),
        options: triggerPhrases.map((phrase) => ({ label: phrase, value: phrase })),
      },
    ];
    return [...metadataOptions, ...embeddingOptions];
  }, [data, currentBaseModel, triggerPhrases, t]);

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
