import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  selectWanComponentSource,
  selectWanT5EncoderModel,
  selectWanTransformerLowNoise,
  selectWanVaeModel,
  wanComponentSourceSelected,
  wanT5EncoderModelSelected,
  wanTransformerLowNoiseSelected,
  wanVaeModelSelected,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useWanDiffusersModels,
  useWanGGUFLowNoiseModels,
  useWanT5EncoderModels,
  useWanVAEModels,
} from 'services/api/hooks/modelsByType';
import type { MainModelConfig, VAEModelConfig, WanT5EncoderModelConfig } from 'services/api/types';

/**
 * Wan 2.2 Transformer (Low Noise) Select
 *
 * Picks the second-expert GGUF transformer for an A14B MoE workflow. Only
 * relevant when the main Wan model is a GGUF — Diffusers A14B already carries
 * both experts in transformer/ and transformer_2/ subfolders.
 */
const ParamWanTransformerLowNoiseSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const value = useAppSelector(selectWanTransformerLowNoise);
  const [modelConfigs, { isLoading }] = useWanGGUFLowNoiseModels();

  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (model) {
        dispatch(wanTransformerLowNoiseSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(wanTransformerLowNoiseSelected(null));
      }
    },
    [dispatch]
  );

  const {
    options,
    value: comboValue,
    onChange,
    noOptionsMessage,
  } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: value,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.wanTransformerLowNoise')}</FormLabel>
      <Combobox
        value={comboValue}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.wanTransformerLowNoisePlaceholder')}
      />
    </FormControl>
  );
});

ParamWanTransformerLowNoiseSelect.displayName = 'ParamWanTransformerLowNoiseSelect';

/**
 * Wan 2.2 Component Source Select
 *
 * Picks a Diffusers Wan model whose VAE and UMT5-XXL encoder will be extracted
 * for the workflow. Required when the main Wan model is a GGUF (since GGUF
 * mains are transformer-only). Ignored for Diffusers mains, which carry their
 * own VAE and encoder.
 */
const ParamWanComponentSourceSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const value = useAppSelector(selectWanComponentSource);
  const [modelConfigs, { isLoading }] = useWanDiffusersModels();

  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (model) {
        dispatch(wanComponentSourceSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(wanComponentSourceSelected(null));
      }
    },
    [dispatch]
  );

  const {
    options,
    value: comboValue,
    onChange,
    noOptionsMessage,
  } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: value,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.wanComponentSource')}</FormLabel>
      <Combobox
        value={comboValue}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.wanComponentSourcePlaceholder')}
      />
    </FormControl>
  );
});

ParamWanComponentSourceSelect.displayName = 'ParamWanComponentSourceSelect';

/**
 * Wan 2.2 Standalone VAE Select
 *
 * Selects a standalone Wan VAE checkpoint. When set, this overrides the VAE
 * provided by the Component Source (or the main Diffusers model).
 */
const ParamWanVaeModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const vaeModel = useAppSelector(selectWanVaeModel);
  const [modelConfigs, { isLoading }] = useWanVAEModels();

  const _onChange = useCallback(
    (model: VAEModelConfig | null) => {
      if (model) {
        dispatch(wanVaeModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(wanVaeModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: vaeModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.wanVae')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.wanVaePlaceholder')}
      />
    </FormControl>
  );
});

ParamWanVaeModelSelect.displayName = 'ParamWanVaeModelSelect';

/**
 * Wan 2.2 Standalone UMT5-XXL Encoder Select
 *
 * Selects a standalone UMT5-XXL encoder. When set, this overrides the encoder
 * provided by the Component Source (or the main Diffusers model).
 */
const ParamWanT5EncoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const encoderModel = useAppSelector(selectWanT5EncoderModel);
  const [modelConfigs, { isLoading }] = useWanT5EncoderModels();

  const _onChange = useCallback(
    (model: WanT5EncoderModelConfig | null) => {
      if (model) {
        dispatch(wanT5EncoderModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(wanT5EncoderModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: encoderModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.wanT5Encoder')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.wanT5EncoderPlaceholder')}
      />
    </FormControl>
  );
});

ParamWanT5EncoderModelSelect.displayName = 'ParamWanT5EncoderModelSelect';

/**
 * Combined Wan 2.2 component selectors (low-noise transformer + standalone
 * VAE + standalone T5 encoder + Component Source).
 *
 * Only relevant for GGUF workflows. Diffusers Wan mains have everything
 * built in; TI2V-5B is a single-expert model with no low-noise pair. Showing
 * these always is fine since they're optional — but the AdvancedSettingsAccordion
 * still gates the render on `isWan` so they don't pollute other tabs.
 */
const ParamWanModelSelects = () => {
  return (
    <>
      <ParamWanTransformerLowNoiseSelect />
      <ParamWanVaeModelSelect />
      <ParamWanT5EncoderModelSelect />
      <ParamWanComponentSourceSelect />
    </>
  );
};

export default memo(ParamWanModelSelects);
