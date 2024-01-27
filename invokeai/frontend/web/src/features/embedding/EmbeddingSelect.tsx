import type { ChakraProps } from '@invoke-ai/ui-library';
import { Combobox, FormControl } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import type { EmbeddingSelectProps } from 'features/embedding/types';
import { t } from 'i18next';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type { TextualInversionModelConfigEntity } from 'services/api/endpoints/models';
import { useGetTextualInversionModelsQuery } from 'services/api/endpoints/models';

const noOptionsMessage = () => t('embedding.noMatchingEmbedding');

export const EmbeddingSelect = memo(({ onSelect, onClose }: EmbeddingSelectProps) => {
  const { t } = useTranslation();

  const currentBaseModel = useAppSelector((s) => s.generation.model?.base_model);

  const getIsDisabled = useCallback(
    (embedding: TextualInversionModelConfigEntity): boolean => {
      const isCompatible = currentBaseModel === embedding.base_model;
      const hasMainModel = Boolean(currentBaseModel);
      return !hasMainModel || !isCompatible;
    },
    [currentBaseModel]
  );
  const { data, isLoading } = useGetTextualInversionModelsQuery();

  const _onChange = useCallback(
    (embedding: TextualInversionModelConfigEntity | null) => {
      if (!embedding) {
        return;
      }
      onSelect(embedding.model_name);
    },
    [onSelect]
  );

  const { options, onChange } = useGroupedModelCombobox({
    modelEntities: data,
    getIsDisabled,
    onChange: _onChange,
  });

  return (
    <FormControl>
      <Combobox
        placeholder={isLoading ? t('common.loading') : t('embedding.addEmbedding')}
        defaultMenuIsOpen
        autoFocus
        value={null}
        options={options}
        noOptionsMessage={noOptionsMessage}
        onChange={onChange}
        onMenuClose={onClose}
        data-testid="add-embedding"
        sx={selectStyles}
      />
    </FormControl>
  );
});

EmbeddingSelect.displayName = 'EmbeddingSelect';

const selectStyles: ChakraProps['sx'] = {
  w: 'full',
};
