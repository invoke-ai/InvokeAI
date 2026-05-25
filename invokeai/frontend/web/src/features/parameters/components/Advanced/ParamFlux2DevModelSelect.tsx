import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  flux2DevMistralEncoderModelSelected,
  flux2DevVaeModelSelected,
  selectFlux2DevMistralEncoderModel,
  selectFlux2DevVaeModel,
  selectMainModelConfig,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useFlux2DevDiffusersModels,
  useFlux2VAEModels,
  useMistralEncoderModels,
} from 'services/api/hooks/modelsByType';
import type { MistralEncoderModelConfig, VAEModelConfig } from 'services/api/types';

/**
 * FLUX.2 [dev] VAE Model Select.
 *
 * Selects the 32-channel AutoencoderKLFlux2 VAE used by FLUX.2 [dev]. This is
 * the same VAE family as FLUX.2 Klein, so the shared FLUX.2 VAE pool applies.
 */
const ParamFlux2DevVaeModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const flux2DevVaeModel = useAppSelector(selectFlux2DevVaeModel);
  const mainModelConfig = useAppSelector(selectMainModelConfig);
  const [modelConfigs, { isLoading }] = useFlux2VAEModels();
  const [diffusersModels] = useFlux2DevDiffusersModels();

  const _onChange = useCallback(
    (model: VAEModelConfig | null) => {
      if (model) {
        dispatch(flux2DevVaeModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(flux2DevVaeModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: flux2DevVaeModel,
    isLoading,
  });

  const hasDiffusersSource = mainModelConfig?.format === 'diffusers' || diffusersModels.length > 0;
  const placeholder = hasDiffusersSource
    ? t('modelManager.flux2DevVaePlaceholder', { defaultValue: 'Auto (from Diffusers source)' })
    : t('modelManager.flux2DevVaeNoModelPlaceholder', { defaultValue: 'Select a FLUX.2 VAE model' });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.flux2DevVae', { defaultValue: 'FLUX.2 [dev] VAE' })}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={placeholder}
      />
    </FormControl>
  );
});

ParamFlux2DevVaeModelSelect.displayName = 'ParamFlux2DevVaeModelSelect';

/**
 * FLUX.2 [dev] Mistral Encoder Model Select.
 *
 * Selects the Mistral Small 3.1 text encoder used by FLUX.2 [dev]. Only needed
 * when the main model is a single-file safetensors or GGUF without a Diffusers
 * companion to extract the encoder from.
 */
const ParamFlux2DevMistralEncoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const mistralEncoderModel = useAppSelector(selectFlux2DevMistralEncoderModel);
  const mainModelConfig = useAppSelector(selectMainModelConfig);
  const [modelConfigs, { isLoading }] = useMistralEncoderModels();
  const [diffusersModels] = useFlux2DevDiffusersModels();

  const _onChange = useCallback(
    (model: MistralEncoderModelConfig | null) => {
      if (model) {
        dispatch(flux2DevMistralEncoderModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(flux2DevMistralEncoderModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: mistralEncoderModel,
    isLoading,
  });

  const hasDiffusersSource = mainModelConfig?.format === 'diffusers' || diffusersModels.length > 0;
  const placeholder = hasDiffusersSource
    ? t('modelManager.flux2DevMistralEncoderPlaceholder', { defaultValue: 'Auto (from Diffusers source)' })
    : t('modelManager.flux2DevMistralEncoderNoModelPlaceholder', {
        defaultValue: 'Select a Mistral text encoder',
      });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>
        {t('modelManager.flux2DevMistralEncoder', { defaultValue: 'FLUX.2 [dev] Mistral Encoder' })}
      </FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={placeholder}
      />
    </FormControl>
  );
});

ParamFlux2DevMistralEncoderModelSelect.displayName = 'ParamFlux2DevMistralEncoderModelSelect';

/**
 * Combined component for FLUX.2 [dev] companion model selection.
 */
const ParamFlux2DevModelSelects = () => {
  return (
    <>
      <ParamFlux2DevVaeModelSelect />
      <ParamFlux2DevMistralEncoderModelSelect />
    </>
  );
};

export default memo(ParamFlux2DevModelSelects);
