import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  animaQwen3EncoderModelSelected,
  animaT5EncoderModelSelected,
  animaVaeModelSelected,
  selectAnimaQwen3EncoderModel,
  selectAnimaT5EncoderModel,
  selectAnimaVaeModel,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useAnimaQwen3EncoderModels, useAnimaVAEModels, useT5EncoderModels } from 'services/api/hooks/modelsByType';
import type { Qwen3EncoderModelConfig, T5EncoderModelConfig, VAEModelConfig } from 'services/api/types';

/**
 * Anima VAE Model Select - uses Anima-base VAE models (QwenImage/Wan 2.1 VAE)
 */
const ParamAnimaVaeModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const animaVaeModel = useAppSelector(selectAnimaVaeModel);
  const [modelConfigs, { isLoading }] = useAnimaVAEModels();

  const _onChange = useCallback(
    (model: VAEModelConfig | null) => {
      if (model) {
        dispatch(animaVaeModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(animaVaeModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: animaVaeModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.animaVae')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.animaVaePlaceholder')}
      />
    </FormControl>
  );
});

ParamAnimaVaeModelSelect.displayName = 'ParamAnimaVaeModelSelect';

/**
 * Anima Qwen3 0.6B Encoder Model Select
 */
const ParamAnimaQwen3EncoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const animaQwen3EncoderModel = useAppSelector(selectAnimaQwen3EncoderModel);
  const [modelConfigs, { isLoading }] = useAnimaQwen3EncoderModels();

  const _onChange = useCallback(
    (model: Qwen3EncoderModelConfig | null) => {
      if (model) {
        dispatch(animaQwen3EncoderModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(animaQwen3EncoderModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: animaQwen3EncoderModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.animaQwen3Encoder')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.animaQwen3EncoderPlaceholder')}
      />
    </FormControl>
  );
});

ParamAnimaQwen3EncoderModelSelect.displayName = 'ParamAnimaQwen3EncoderModelSelect';

/**
 * Anima T5 Encoder Model Select - uses T5-XXL encoder models (tokenizer submodel used for Anima)
 */
const ParamAnimaT5EncoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const animaT5EncoderModel = useAppSelector(selectAnimaT5EncoderModel);
  const [modelConfigs, { isLoading }] = useT5EncoderModels();

  const _onChange = useCallback(
    (model: T5EncoderModelConfig | null) => {
      if (model) {
        dispatch(animaT5EncoderModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(animaT5EncoderModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: animaT5EncoderModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.animaT5Encoder')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.animaT5EncoderPlaceholder')}
      />
    </FormControl>
  );
});

ParamAnimaT5EncoderModelSelect.displayName = 'ParamAnimaT5EncoderModelSelect';

/**
 * Combined component for Anima model selection (VAE + Qwen3 Encoder + T5 Encoder)
 */
const ParamAnimaModelSelect = () => {
  return (
    <>
      <ParamAnimaVaeModelSelect />
      <ParamAnimaQwen3EncoderModelSelect />
      <ParamAnimaT5EncoderModelSelect />
    </>
  );
};

export default memo(ParamAnimaModelSelect);
