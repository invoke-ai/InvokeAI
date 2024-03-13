import { CustomSelect, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useModelCustomSelect } from 'common/hooks/useModelCustomSelect';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { modelSelected } from 'features/parameters/store/actions';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { NON_REFINER_BASE_MODELS } from 'services/api/constants';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';
import type { MainModelConfig } from 'services/api/types';

const selectModel = createMemoizedSelector(selectGenerationSlice, (generation) => generation.model);

const ParamMainModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectedModel = useAppSelector(selectModel);
  console.log({ selectedModel });
  const { data, isLoading } = useGetMainModelsQuery(NON_REFINER_BASE_MODELS);

  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (!model) {
        return;
      }
      try {
        dispatch(modelSelected(zModelIdentifierField.parse(model)));
      } catch {
        // no-op
      }
    },
    [dispatch]
  );

  const { items, selectedItem, onChange, placeholder } = useModelCustomSelect({
    data,
    isLoading,
    selectedModel,
    onChange: _onChange,
  });

  return (
    <FormControl isDisabled={!items.length} isInvalid={!selectedItem || !items.length}>
      <InformationalPopover feature="paramModel">
        <FormLabel>{t('modelManager.model')}</FormLabel>
      </InformationalPopover>
      <CustomSelect
        key={items.length}
        selectedItem={selectedItem}
        placeholder={placeholder}
        items={items}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamMainModelSelect);
