import type { ChakraProps } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { loraAdded, selectLoraSlice } from 'features/lora/store/loraSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useLoRAModels } from 'services/api/hooks/modelsByType';
import type { LoRAModelConfig } from 'services/api/types';

const selectAddedLoRAs = createMemoizedSelector(selectLoraSlice, (lora) => lora.loras);

const LoRASelect = () => {
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useLoRAModels();
  const { t } = useTranslation();
  const addedLoRAs = useAppSelector(selectAddedLoRAs);
  const currentBaseModel = useAppSelector((s) => s.generation.model?.base);

  const getIsDisabled = (lora: LoRAModelConfig): boolean => {
    const isCompatible = currentBaseModel === lora.base;
    const isAdded = Boolean(addedLoRAs[lora.key]);
    const hasMainModel = Boolean(currentBaseModel);
    return !hasMainModel || !isCompatible || isAdded;
  };

  const _onChange = useCallback(
    (lora: LoRAModelConfig | null) => {
      if (!lora) {
        return;
      }
      dispatch(loraAdded(lora));
    },
    [dispatch]
  );

  const { options, onChange } = useGroupedModelCombobox({
    modelConfigs,
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
      <InformationalPopover feature="lora">
        <FormLabel>{t('models.concepts')} </FormLabel>
      </InformationalPopover>
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
