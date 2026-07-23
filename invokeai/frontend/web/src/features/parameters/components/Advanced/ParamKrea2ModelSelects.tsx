import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  krea2Qwen3VlEncoderModelSelected,
  krea2VaeModelSelected,
  selectKrea2Qwen3VlEncoderModel,
  selectKrea2VaeModel,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useAnimaVAEModels, useQwen3VLEncoderModels, useQwenImageVAEModels } from 'services/api/hooks/modelsByType';
import type { Qwen3VLEncoderModelConfig, VAEModelConfig } from 'services/api/types';

/**
 * Krea-2 VAE Model Select - Krea-2 uses the Qwen-Image VAE (16-channel). Optional override used when the
 * transformer is a single-file checkpoint/GGUF without a bundled VAE.
 */
const ParamKrea2VaeModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const krea2VaeModel = useAppSelector(selectKrea2VaeModel);
  // Krea-2 / Qwen-Image / Anima share the identical AutoencoderKLQwenImage VAE. A standalone
  // qwen_image_vae.safetensors is classified as either base (the weights are indistinguishable), so
  // accept both here.
  const [qwenImageVaes, { isLoading: isLoadingQwen }] = useQwenImageVAEModels();
  const [animaVaes, { isLoading: isLoadingAnima }] = useAnimaVAEModels();
  const modelConfigs = useMemo(() => [...qwenImageVaes, ...animaVaes], [qwenImageVaes, animaVaes]);
  const isLoading = isLoadingQwen || isLoadingAnima;

  const _onChange = useCallback(
    (model: VAEModelConfig | null) => {
      dispatch(krea2VaeModelSelected(model ? zModelIdentifierField.parse(model) : null));
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: krea2VaeModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.krea2Vae')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.krea2VaePlaceholder')}
      />
    </FormControl>
  );
});

ParamKrea2VaeModelSelect.displayName = 'ParamKrea2VaeModelSelect';

/**
 * Krea-2 Qwen3-VL Encoder Model Select - optional standalone encoder used when the transformer is a
 * single-file checkpoint/GGUF without a bundled encoder.
 */
const ParamKrea2Qwen3VlEncoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const krea2Qwen3VlEncoderModel = useAppSelector(selectKrea2Qwen3VlEncoderModel);
  const [modelConfigs, { isLoading }] = useQwen3VLEncoderModels();

  const _onChange = useCallback(
    (model: Qwen3VLEncoderModelConfig | null) => {
      dispatch(krea2Qwen3VlEncoderModelSelected(model ? zModelIdentifierField.parse(model) : null));
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: krea2Qwen3VlEncoderModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.krea2Qwen3VlEncoder')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.krea2Qwen3VlEncoderPlaceholder')}
      />
    </FormControl>
  );
});

ParamKrea2Qwen3VlEncoderModelSelect.displayName = 'ParamKrea2Qwen3VlEncoderModelSelect';

/**
 * Combined component for Krea-2 standalone submodel selection (VAE + Qwen3-VL encoder).
 */
const ParamKrea2ModelSelects = () => {
  return (
    <>
      <ParamKrea2VaeModelSelect />
      <ParamKrea2Qwen3VlEncoderModelSelect />
    </>
  );
};

export default memo(ParamKrea2ModelSelects);
