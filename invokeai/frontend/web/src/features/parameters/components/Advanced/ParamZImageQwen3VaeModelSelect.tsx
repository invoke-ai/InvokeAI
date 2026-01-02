import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  selectZImageQwen3EncoderModel,
  selectZImageQwen3SourceModel,
  selectZImageVaeModel,
  zImageQwen3EncoderModelSelected,
  zImageQwen3SourceModelSelected,
  zImageVaeModelSelected,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useFluxVAEModels, useQwen3EncoderModels, useZImageDiffusersModels } from 'services/api/hooks/modelsByType';
import type { MainModelConfig, Qwen3EncoderModelConfig, VAEModelConfig } from 'services/api/types';

/**
 * Z-Image VAE Model Select - uses FLUX VAE models
 * Disabled when Qwen3 Source is selected (mutually exclusive)
 */
const ParamZImageVaeModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const zImageVaeModel = useAppSelector(selectZImageVaeModel);
  const zImageQwen3SourceModel = useAppSelector(selectZImageQwen3SourceModel);
  const [modelConfigs, { isLoading }] = useFluxVAEModels();

  // Disable when Qwen3 Source is selected
  const isDisabled = zImageQwen3SourceModel !== null;

  const _onChange = useCallback(
    (model: VAEModelConfig | null) => {
      if (model) {
        dispatch(zImageVaeModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(zImageVaeModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: zImageVaeModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2} isDisabled={isDisabled}>
      <FormLabel m={0}>{t('modelManager.zImageVae')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        isDisabled={isDisabled}
        placeholder={t('modelManager.zImageVaePlaceholder')}
      />
    </FormControl>
  );
});

ParamZImageVaeModelSelect.displayName = 'ParamZImageVaeModelSelect';

/**
 * Z-Image Qwen3 Encoder Model Select
 * Disabled when Qwen3 Source is selected (mutually exclusive)
 */
const ParamZImageQwen3EncoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const zImageQwen3EncoderModel = useAppSelector(selectZImageQwen3EncoderModel);
  const zImageQwen3SourceModel = useAppSelector(selectZImageQwen3SourceModel);
  const [modelConfigs, { isLoading }] = useQwen3EncoderModels();

  // Disable when Qwen3 Source is selected
  const isDisabled = zImageQwen3SourceModel !== null;

  const _onChange = useCallback(
    (model: Qwen3EncoderModelConfig | null) => {
      if (model) {
        dispatch(zImageQwen3EncoderModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(zImageQwen3EncoderModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: zImageQwen3EncoderModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2} isDisabled={isDisabled}>
      <FormLabel m={0}>{t('modelManager.zImageQwen3Encoder')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        isDisabled={isDisabled}
        placeholder={t('modelManager.zImageQwen3EncoderPlaceholder')}
      />
    </FormControl>
  );
});

ParamZImageQwen3EncoderModelSelect.displayName = 'ParamZImageQwen3EncoderModelSelect';

/**
 * Z-Image Qwen3 Source Model Select - Diffusers Z-Image models for fallback
 * Disabled when VAE or Qwen3 Encoder is selected (mutually exclusive)
 */
const ParamZImageQwen3SourceModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const zImageQwen3SourceModel = useAppSelector(selectZImageQwen3SourceModel);
  const zImageVaeModel = useAppSelector(selectZImageVaeModel);
  const zImageQwen3EncoderModel = useAppSelector(selectZImageQwen3EncoderModel);
  const [modelConfigs, { isLoading }] = useZImageDiffusersModels();

  // Disable when VAE or Qwen3 Encoder is selected
  const isDisabled = zImageVaeModel !== null || zImageQwen3EncoderModel !== null;

  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (model) {
        dispatch(zImageQwen3SourceModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(zImageQwen3SourceModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: zImageQwen3SourceModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2} isDisabled={isDisabled}>
      <FormLabel m={0}>{t('modelManager.zImageQwen3Source')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        isDisabled={isDisabled}
        placeholder={t('modelManager.zImageQwen3SourcePlaceholder')}
      />
    </FormControl>
  );
});

ParamZImageQwen3SourceModelSelect.displayName = 'ParamZImageQwen3SourceModelSelect';

/**
 * Combined component for Z-Image model selection
 */
const ParamZImageModelSelects = () => {
  return (
    <>
      <ParamZImageVaeModelSelect />
      <ParamZImageQwen3EncoderModelSelect />
      <ParamZImageQwen3SourceModelSelect />
    </>
  );
};

export default memo(ParamZImageModelSelects);
