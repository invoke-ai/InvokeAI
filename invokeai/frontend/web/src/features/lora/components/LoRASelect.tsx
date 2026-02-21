import { FormControl, FormLabel } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { GroupStatusMap } from 'common/components/Picker/Picker';
import { loraAdded, selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectBase, selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useLoRAModels } from 'services/api/hooks/modelsByType';
import type { LoRAModelConfig } from 'services/api/types';

const selectLoRAModelKeys = createMemoizedSelector(selectLoRAsSlice, ({ loras }) =>
  loras.map(({ model }) => model.key)
);

const LoRASelect = () => {
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useLoRAModels();
  const { t } = useTranslation();
  const addedLoRAModelKeys = useAppSelector(selectLoRAModelKeys);

  const currentBaseModel = useAppSelector(selectBase);
  const currentMainModelConfig = useAppSelector(selectMainModelConfig);

  // Filter to only show compatible LoRAs (by base model and variant)
  const compatibleLoRAs = useMemo(() => {
    if (!currentBaseModel) {
      return EMPTY_ARRAY;
    }
    return modelConfigs.filter((model) => {
      if (model.base !== currentBaseModel) {
        return false;
      }
      // For models with variant support: filter by variant when both main model and LoRA have variant info.
      // LoRAs with no variant (null) are always shown (compatible with all variants).
      if (
        currentMainModelConfig &&
        'variant' in currentMainModelConfig &&
        currentMainModelConfig.variant &&
        'variant' in model &&
        model.variant
      ) {
        return model.variant === currentMainModelConfig.variant;
      }
      return true;
    });
  }, [modelConfigs, currentBaseModel, currentMainModelConfig]);

  const getIsDisabled = useCallback(
    (model: LoRAModelConfig): boolean => {
      const isAdded = addedLoRAModelKeys.includes(model.key);
      return isAdded;
    },
    [addedLoRAModelKeys]
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
      return currentBaseModel ? t('models.noCompatibleLoRAs') : t('models.selectModel');
    }

    return t('models.addLora');
  }, [isLoading, compatibleLoRAs.length, currentBaseModel, t]);

  // Calculate initial group states to default to the current base model architecture
  const initialGroupStates = useMemo(() => {
    if (!currentBaseModel) {
      return undefined;
    }

    // Return a map with only the current base model group enabled
    return { [currentBaseModel]: true } satisfies GroupStatusMap;
  }, [currentBaseModel]);

  return (
    <FormControl gap={2}>
      <InformationalPopover feature="lora">
        <FormLabel>{t('models.concepts')} </FormLabel>
      </InformationalPopover>
      <ModelPicker
        pickerId="lora-select"
        modelConfigs={compatibleLoRAs}
        onChange={onChange}
        grouped={false}
        selectedModelConfig={undefined}
        allowEmpty
        placeholder={placeholder}
        getIsOptionDisabled={getIsDisabled}
        initialGroupStates={initialGroupStates}
        noOptionsText={currentBaseModel ? t('models.noCompatibleLoRAs') : t('models.selectModel')}
      />
    </FormControl>
  );
};

export default memo(LoRASelect);
