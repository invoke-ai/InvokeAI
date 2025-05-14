import { Flex, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { loraAdded, selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectBase } from 'features/controlLayers/store/paramsSlice';
import { UseDefaultSettingsButton } from 'features/parameters/components/MainModel/UseDefaultSettingsButton';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useLoRAModels } from 'services/api/hooks/modelsByType';
import type { LoRAModelConfig } from 'services/api/types';

const selectLoRAs = createSelector(selectLoRAsSlice, (loras) => loras.loras);

export const LoRAModelPicker = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const currentBaseModel = useAppSelector(selectBase);
  const addedLoRAs = useAppSelector(selectLoRAs);

  const getIsOptionDisabled = useCallback(
    (model: LoRAModelConfig): boolean => {
      const isCompatible = currentBaseModel === model.base;
      const isAdded = Boolean(addedLoRAs.find((lora) => lora.model.key === model.key));
      return !isCompatible || isAdded;
    },
    [addedLoRAs, currentBaseModel]
  );

  const loraFilter = useCallback(
    (loraConfig: LoRAModelConfig) => {
      if (!currentBaseModel) {
        return true;
      }
      return currentBaseModel === loraConfig.base;
    },
    [currentBaseModel]
  );
  const [modelConfigs] = useLoRAModels();
  const onChange = useCallback(
    (loraConfig: LoRAModelConfig) => {
      dispatch(loraAdded({ model: loraConfig }));
    },
    [dispatch]
  );

  return (
    <Flex alignItems="center" gap={2}>
      <InformationalPopover feature="lora">
        <FormLabel>{t('models.concepts')} </FormLabel>
      </InformationalPopover>
      <ModelPicker
        modelConfigs={modelConfigs}
        selectedModelConfig={undefined}
        onChange={onChange}
        placeholder={t('models.addLora')}
        getIsOptionDisabled={getIsOptionDisabled}
        defaultEnabledGroups={currentBaseModel ? [currentBaseModel] : undefined}
        allowEmpty
        grouped
      />
      <UseDefaultSettingsButton />
    </Flex>
  );
});
LoRAModelPicker.displayName = 'LoRAModelPicker';
