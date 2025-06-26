import { FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useRelatedGroupedModelCombobox } from 'common/hooks/useRelatedGroupedModelCombobox';
import { loraAdded, selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectBase } from 'features/controlLayers/store/paramsSlice';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { API_BASE_MODELS } from 'features/parameters/types/constants';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useLoRAModels } from 'services/api/hooks/modelsByType';
import type { LoRAModelConfig } from 'services/api/types';

const selectLoRAs = createSelector(selectLoRAsSlice, (loras) => loras.loras);

const LoRASelect = () => {
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useLoRAModels();
  const { t } = useTranslation();
  const addedLoRAs = useAppSelector(selectLoRAs);
  const currentBaseModel = useAppSelector(selectBase);

  const getIsDisabled = useCallback(
    (model: LoRAModelConfig): boolean => {
      const isCompatible = currentBaseModel === model.base;
      const isAdded = Boolean(addedLoRAs.find((lora) => lora.model.key === model.key));
      const hasMainModel = Boolean(currentBaseModel);
      return !hasMainModel || !isCompatible || isAdded;
    },
    [addedLoRAs, currentBaseModel]
  );

  const onChange = useCallback(
    (model: LoRAModelConfig | null) => {
      if (!model) {
        return;
      }
      dispatch(loraAdded({ model }));
    },
    [dispatch]
  );

  const { options } = useRelatedGroupedModelCombobox({
    modelConfigs,
    getIsDisabled,
    onChange,
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

  // Calculate initial group states to default to the current base model architecture
  const initialGroupStates = useMemo(() => {
    if (!currentBaseModel) {
      return undefined;
    }

    // Determine the group ID for the current base model
    const groupId = API_BASE_MODELS.includes(currentBaseModel) ? 'api' : currentBaseModel;

    // Return a map with only the current base model group enabled
    return { [groupId]: true };
  }, [currentBaseModel]);

  return (
    <FormControl gap={2}>
      <InformationalPopover feature="lora">
        <FormLabel>{t('models.concepts')} </FormLabel>
      </InformationalPopover>
      <ModelPicker
        modelConfigs={modelConfigs}
        onChange={onChange}
        grouped
        selectedModelConfig={undefined}
        allowEmpty
        placeholder={placeholder}
        getIsOptionDisabled={getIsDisabled}
        noOptionsText={t('models.noLoRAsInstalled')}
        initialGroupStates={initialGroupStates}
      />
    </FormControl>
  );
};

export default memo(LoRASelect);
