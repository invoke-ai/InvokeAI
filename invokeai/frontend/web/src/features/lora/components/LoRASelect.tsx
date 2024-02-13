import type { ChakraProps } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { loraAdded, selectLoraSlice } from 'features/lora/store/loraSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { LoRAModelConfigEntity } from 'services/api/endpoints/models';
import { useGetLoRAModelsQuery } from 'services/api/endpoints/models';

const selectAddedLoRAs = createMemoizedSelector(selectLoraSlice, (lora) => lora.loras);

const LoRASelect = () => {
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetLoRAModelsQuery();
  const { t } = useTranslation();
  const addedLoRAs = useAppSelector(selectAddedLoRAs);
  const currentBaseModel = useAppSelector((s) => s.generation.model?.base_model);

  const getIsDisabled = (lora: LoRAModelConfigEntity): boolean => {
    const isCompatible = currentBaseModel === lora.base_model;
    const isAdded = Boolean(addedLoRAs[lora.id]);
    const hasMainModel = Boolean(currentBaseModel);
    return !hasMainModel || !isCompatible || isAdded;
  };

  const _onChange = useCallback(
    (lora: LoRAModelConfigEntity | null) => {
      if (!lora) {
        return;
      }
      dispatch(loraAdded(lora));
    },
    [dispatch]
  );

  const { options, onChange } = useGroupedModelCombobox({
    modelEntities: data,
    getIsDisabled,
    onChange: _onChange,
  });

  const placeholder = useMemo(() => {
    if (isLoading) {
      return t('common.loading');
    }

    if (options.length === 0) {
      return t('models.noLoRAsInstalled');
    }

    return t('models.addLora');
  }, [isLoading, options.length, t]);

  const noOptionsMessage = useCallback(() => t('models.noMatchingLoRAs'), [t]);

  return (
    <FormControl isDisabled={!options.length}>
      <FormLabel>{t('models.lora')} </FormLabel>
      <Combobox
        placeholder={placeholder}
        value={null}
        options={options}
        noOptionsMessage={noOptionsMessage}
        onChange={onChange}
        data-testid="add-lora"
        sx={selectStyles}
      />
    </FormControl>
  );
};

export default memo(LoRASelect);

const selectStyles: ChakraProps['sx'] = {
  w: 'full',
};
