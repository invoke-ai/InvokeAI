import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  qwenImageComponentSourceSelected,
  qwenImageQwenVLEncoderModelSelected,
  qwenImageVaeModelSelected,
  selectQwenImageComponentSource,
  selectQwenImageQwenVLEncoderModel,
  selectQwenImageVaeModel,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useQwenImageDiffusersModels,
  useQwenImageVAEModels,
  useQwenVLEncoderModels,
} from 'services/api/hooks/modelsByType';
import type { MainModelConfig, QwenVLEncoderModelConfig, VAEModelConfig } from 'services/api/types';

/**
 * Qwen Image Standalone VAE Select
 *
 * Selects a standalone Qwen Image VAE checkpoint. When set, this overrides the
 * VAE provided by the Component Source (or the main Diffusers model).
 */
const ParamQwenImageVaeModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const vaeModel = useAppSelector(selectQwenImageVaeModel);
  const [modelConfigs, { isLoading }] = useQwenImageVAEModels();

  const _onChange = useCallback(
    (model: VAEModelConfig | null) => {
      if (model) {
        dispatch(qwenImageVaeModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(qwenImageVaeModelSelected(null));
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
      <FormLabel m={0}>{t('modelManager.qwenImageVae')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.qwenImageVaePlaceholder')}
      />
    </FormControl>
  );
});

ParamQwenImageVaeModelSelect.displayName = 'ParamQwenImageVaeModelSelect';

/**
 * Qwen Image Standalone Qwen2.5-VL Encoder Select
 *
 * Selects a standalone Qwen2.5-VL encoder. When set, this overrides the encoder
 * provided by the Component Source (or the main Diffusers model).
 */
const ParamQwenImageQwenVLEncoderSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const encoderModel = useAppSelector(selectQwenImageQwenVLEncoderModel);
  const [modelConfigs, { isLoading }] = useQwenVLEncoderModels();

  const _onChange = useCallback(
    (model: QwenVLEncoderModelConfig | null) => {
      if (model) {
        dispatch(qwenImageQwenVLEncoderModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(qwenImageQwenVLEncoderModelSelected(null));
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
      <FormLabel m={0}>{t('modelManager.qwenImageQwenVLEncoder')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.qwenImageQwenVLEncoderPlaceholder')}
      />
    </FormControl>
  );
});

ParamQwenImageQwenVLEncoderSelect.displayName = 'ParamQwenImageQwenVLEncoderSelect';

/**
 * Qwen Image Edit Component Source Model Select
 *
 * Selects a Diffusers Qwen Image Edit model to provide the VAE and text encoder
 * when using a GGUF quantized transformer.
 */
const ParamQwenImageComponentSourceSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const componentSource = useAppSelector(selectQwenImageComponentSource);
  const [modelConfigs, { isLoading }] = useQwenImageDiffusersModels();

  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (model) {
        dispatch(qwenImageComponentSourceSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(qwenImageComponentSourceSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: componentSource,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.qwenImageComponentSource')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.qwenImageComponentSourcePlaceholder')}
      />
    </FormControl>
  );
});

ParamQwenImageComponentSourceSelect.displayName = 'ParamQwenImageComponentSourceSelect';

/**
 * Combined Qwen Image model component selectors (standalone VAE + Component Source).
 */
const ParamQwenImageModelSelects = () => {
  return (
    <>
      <ParamQwenImageVaeModelSelect />
      <ParamQwenImageQwenVLEncoderSelect />
      <ParamQwenImageComponentSourceSelect />
    </>
  );
};

export default memo(ParamQwenImageModelSelects);
