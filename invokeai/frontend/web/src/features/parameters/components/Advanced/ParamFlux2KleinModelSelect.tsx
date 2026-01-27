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
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useFlux2VAEModels, useQwen3EncoderModels } from 'services/api/hooks/modelsByType';
import type { Qwen3EncoderModelConfig, VAEModelConfig } from 'services/api/types';

/**
 * FLUX.2 Klein VAE Model Select
 * Selects a FLUX.2 VAE model (32-channel AutoencoderKLFlux2)
 */
const ParamFlux2KleinVaeModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const kleinVaeModel = useAppSelector(selectKleinVaeModel);
  const [modelConfigs, { isLoading }] = useFlux2VAEModels();

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

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.flux2KleinVae')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.flux2KleinVaePlaceholder')}
      />
    </FormControl>
  );
});

ParamFlux2KleinVaeModelSelect.displayName = 'ParamFlux2KleinVaeModelSelect';

/**
 * Maps FLUX.2 Klein variants to compatible Qwen3 encoder variants
 */
const KLEIN_TO_QWEN3_VARIANT_MAP: Record<string, string> = {
  klein_4b: 'qwen3_4b',
  klein_9b: 'qwen3_8b',
  klein_9b_base: 'qwen3_8b',
};

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

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.flux2KleinQwen3Encoder')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.flux2KleinQwen3EncoderPlaceholder')}
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
