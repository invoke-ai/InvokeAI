import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  kleinQwen3EncoderModelSelected,
  kleinVaeModelSelected,
  selectKleinQwen3EncoderModel,
  selectKleinVaeModel,
  selectMainModelConfig,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { isFlux2KleinQwen3Compatible, KLEIN_TO_QWEN3_VARIANT_MAP } from 'features/parameters/util/flux2Klein';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useFlux2DiffusersModels, useFlux2VAEModels, useQwen3EncoderModels } from 'services/api/hooks/modelsByType';
import type { Qwen3EncoderModelConfig, VAEModelConfig } from 'services/api/types';

/**
 * FLUX.2 Klein VAE Model Select
 * Selects a FLUX.2 VAE model (32-channel AutoencoderKLFlux2)
 */
const ParamFlux2KleinVaeModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const kleinVaeModel = useAppSelector(selectKleinVaeModel);
  const mainModelConfig = useAppSelector(selectMainModelConfig);
  const [modelConfigs, { isLoading }] = useFlux2VAEModels();
  const [diffusersModels] = useFlux2DiffusersModels();

  const _onChange = useCallback(
    (model: VAEModelConfig | null) => {
      if (model) {
        dispatch(kleinVaeModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(kleinVaeModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: kleinVaeModel,
    isLoading,
  });

  const hasDiffusersSource = mainModelConfig?.format === 'diffusers' || diffusersModels.length > 0;
  const placeholder = hasDiffusersSource
    ? t('modelManager.flux2KleinVaePlaceholder')
    : t('modelManager.flux2KleinVaeNoModelPlaceholder');

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.flux2KleinVae')}</FormLabel>
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

ParamFlux2KleinVaeModelSelect.displayName = 'ParamFlux2KleinVaeModelSelect';

/**
 * FLUX.2 Klein Qwen3 Encoder Model Select
 * Selects a Qwen3 text encoder model for FLUX.2 Klein
 * Only shows encoders compatible with the selected Klein model variant
 */
const ParamFlux2KleinQwen3EncoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const kleinQwen3EncoderModel = useAppSelector(selectKleinQwen3EncoderModel);
  const mainModelConfig = useAppSelector(selectMainModelConfig);
  const [allModelConfigs, { isLoading }] = useQwen3EncoderModels();
  const [diffusersModels] = useFlux2DiffusersModels();

  // Filter Qwen3 encoders based on the main model's variant
  const modelConfigs = useMemo(() => {
    if (!mainModelConfig || !('variant' in mainModelConfig) || !mainModelConfig.variant) {
      return allModelConfigs;
    }

    const requiredQwen3Variant = KLEIN_TO_QWEN3_VARIANT_MAP[mainModelConfig.variant];
    if (!requiredQwen3Variant) {
      return allModelConfigs;
    }

    return allModelConfigs.filter((config) => config.variant === requiredQwen3Variant);
  }, [allModelConfigs, mainModelConfig]);

  const _onChange = useCallback(
    (model: Qwen3EncoderModelConfig | null) => {
      if (model) {
        dispatch(kleinQwen3EncoderModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(kleinQwen3EncoderModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: kleinQwen3EncoderModel,
    isLoading,
  });

  // Qwen3 encoder requires a Qwen3-compatible diffusers model (variants that share the same Qwen3 encoder).
  const hasMatchingDiffusersSource =
    mainModelConfig?.format === 'diffusers' ||
    diffusersModels.some(
      (m) =>
        'variant' in m &&
        mainModelConfig &&
        'variant' in mainModelConfig &&
        isFlux2KleinQwen3Compatible(m.variant, mainModelConfig.variant)
    );
  const placeholder = hasMatchingDiffusersSource
    ? t('modelManager.flux2KleinQwen3EncoderPlaceholder')
    : t('modelManager.flux2KleinQwen3EncoderNoModelPlaceholder');

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.flux2KleinQwen3Encoder')}</FormLabel>
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

ParamFlux2KleinQwen3EncoderModelSelect.displayName = 'ParamFlux2KleinQwen3EncoderModelSelect';

/**
 * Combined component for FLUX.2 Klein model selection
 */
const ParamFlux2KleinModelSelects = () => {
  return (
    <>
      <ParamFlux2KleinVaeModelSelect />
      <ParamFlux2KleinQwen3EncoderModelSelect />
    </>
  );
};

export default memo(ParamFlux2KleinModelSelects);
