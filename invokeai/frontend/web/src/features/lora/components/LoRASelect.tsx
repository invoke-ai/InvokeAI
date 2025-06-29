import { FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { GroupStatusMap } from 'common/components/Picker/Picker';
import { uniq } from 'es-toolkit/compat';
import { loraAdded, selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { API_BASE_MODELS } from 'features/parameters/types/constants';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetRelatedModelIdsBatchQuery } from 'services/api/endpoints/modelRelationships';
import { useLoRAModels } from 'services/api/hooks/modelsByType';
import type { LoRAModelConfig } from 'services/api/types';

const selectLoRAs = createSelector(selectLoRAsSlice, (loras) => loras.loras);

const selectSelectedModelKeys = createSelector(selectParamsSlice, selectLoRAsSlice, (params, loras) => {
  const keys: string[] = [];
  const main = params.model;
  const vae = params.vae;
  const refiner = params.refinerModel;
  const controlnet = params.controlLora;

  if (main) {
    keys.push(main.key);
  }
  if (vae) {
    keys.push(vae.key);
  }
  if (refiner) {
    keys.push(refiner.key);
  }
  if (controlnet) {
    keys.push(controlnet.key);
  }
  for (const { model } of loras.loras) {
    keys.push(model.key);
  }

  return uniq(keys);
});

const LoRASelect = () => {
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useLoRAModels();
  const { t } = useTranslation();
  const addedLoRAs = useAppSelector(selectLoRAs);
  const selectedKeys = useAppSelector(selectSelectedModelKeys);

  const { relatedKeys } = useGetRelatedModelIdsBatchQuery(selectedKeys, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { relatedKeys: EMPTY_ARRAY };
      }
      return { relatedKeys: data };
    },
  });

  const currentBaseModel = useAppSelector((state) => state.params.model?.base);

  // Filter to only show compatible LoRAs
  const compatibleLoRAs = useMemo(() => {
    if (!currentBaseModel) {
      return [];
    }
    return modelConfigs.filter((model) => model.base === currentBaseModel);
  }, [modelConfigs, currentBaseModel]);

  const getIsDisabled = useCallback(
    (model: LoRAModelConfig): boolean => {
      const isAdded = Boolean(addedLoRAs.find((lora) => lora.model.key === model.key));
      return isAdded;
    },
    [addedLoRAs]
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

  const placeholder = useMemo(() => {
    if (isLoading) {
      return t('common.loading');
    }

    if (compatibleLoRAs.length === 0) {
      return currentBaseModel ? t('models.noCompatibleLoRAs') : t('models.selectModelFirst');
    }

    return t('models.addLora');
  }, [isLoading, compatibleLoRAs.length, currentBaseModel, t]);

  // Calculate initial group states to default to the current base model architecture
  const initialGroupStates = useMemo(() => {
    if (!currentBaseModel) {
      return undefined;
    }

    // Determine the group ID for the current base model
    const groupId = API_BASE_MODELS.includes(currentBaseModel) ? 'api' : currentBaseModel;

    // Return a map with only the current base model group enabled
    return { [groupId]: true } satisfies GroupStatusMap;
  }, [currentBaseModel]);

  return (
    <FormControl gap={2}>
      <InformationalPopover feature="lora">
        <FormLabel>{t('models.concepts')} </FormLabel>
      </InformationalPopover>
      <ModelPicker
        modelConfigs={compatibleLoRAs}
        onChange={onChange}
        grouped={false}
        relatedModelKeys={relatedKeys}
        selectedModelConfig={undefined}
        allowEmpty
        placeholder={placeholder}
        getIsOptionDisabled={getIsDisabled}
        initialGroupStates={initialGroupStates}
        noOptionsText={currentBaseModel ? t('models.noCompatibleLoRAs') : t('models.selectModelFirst')}
      />
    </FormControl>
  );
};

export default memo(LoRASelect);
