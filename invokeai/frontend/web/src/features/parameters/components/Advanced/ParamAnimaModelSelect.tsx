import {
  Box,
  Combobox,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormHelperText,
  FormLabel,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  animaLLLiteModelSelected,
  animaLLLiteWeightChanged,
  animaQwen3EncoderModelSelected,
  animaVaeModelSelected,
  selectAnimaLLLiteModel,
  selectAnimaLLLiteWeight,
  selectAnimaQwen3EncoderModel,
  selectAnimaVaeModel,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useAnimaControlNetModels,
  useAnimaQwen3EncoderModels,
  useAnimaVAEModels,
} from 'services/api/hooks/modelsByType';
import type { ControlNetModelConfig, Qwen3EncoderModelConfig, VAEModelConfig } from 'services/api/types';

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
 * Anima ControlNet-LLLite Inpaint Adapter Model Select (optional)
 */
const ParamAnimaLLLiteModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const animaLLLiteModel = useAppSelector(selectAnimaLLLiteModel);
  const [modelConfigs, { isLoading }] = useAnimaControlNetModels();

  const _onChange = useCallback(
    (model: ControlNetModelConfig | null) => {
      if (model) {
        dispatch(animaLLLiteModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(animaLLLiteModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: animaLLLiteModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} orientation="vertical" gap={1}>
      <Flex w="full" alignItems="center" gap={2}>
        <FormLabel m={0}>{t('modelManager.animaInpaintAdapter')}</FormLabel>
        <Box minW={0} flexGrow={1}>
          <Combobox
            value={value}
            options={options}
            onChange={onChange}
            noOptionsMessage={noOptionsMessage}
            isClearable
            placeholder={t('modelManager.animaInpaintAdapterPlaceholder')}
          />
        </Box>
      </Flex>
      <FormHelperText m={0}>{t('modelManager.animaInpaintAdapterHelper')}</FormHelperText>
    </FormControl>
  );
});

ParamAnimaLLLiteModelSelect.displayName = 'ParamAnimaLLLiteModelSelect';

const WEIGHT_CONSTRAINTS = {
  initial: 1,
  sliderMin: 0,
  sliderMax: 2,
  numberInputMin: -10,
  numberInputMax: 10,
  fineStep: 0.01,
  coarseStep: 0.05,
};

const weightMarks = [0, 1, 2];
const formatWeight = (v: number) => v.toFixed(2);

/**
 * Anima ControlNet-LLLite Inpaint Adapter Weight
 */
const ParamAnimaLLLiteWeight = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const animaLLLiteWeight = useAppSelector(selectAnimaLLLiteWeight);

  const onChange = useCallback(
    (v: number) => {
      dispatch(animaLLLiteWeightChanged(v));
    },
    [dispatch]
  );

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.animaInpaintAdapterWeight')}</FormLabel>
      <CompositeSlider
        value={animaLLLiteWeight}
        onChange={onChange}
        defaultValue={WEIGHT_CONSTRAINTS.initial}
        min={WEIGHT_CONSTRAINTS.sliderMin}
        max={WEIGHT_CONSTRAINTS.sliderMax}
        step={WEIGHT_CONSTRAINTS.coarseStep}
        fineStep={WEIGHT_CONSTRAINTS.fineStep}
        marks={weightMarks}
        formatValue={formatWeight}
      />
      <CompositeNumberInput
        value={animaLLLiteWeight}
        onChange={onChange}
        defaultValue={WEIGHT_CONSTRAINTS.initial}
        min={WEIGHT_CONSTRAINTS.numberInputMin}
        max={WEIGHT_CONSTRAINTS.numberInputMax}
        step={WEIGHT_CONSTRAINTS.coarseStep}
        fineStep={WEIGHT_CONSTRAINTS.fineStep}
        maxW={20}
      />
    </FormControl>
  );
});

ParamAnimaLLLiteWeight.displayName = 'ParamAnimaLLLiteWeight';

/**
 * Combined component for Anima model selection (VAE + Qwen3 Encoder + optional Inpaint Adapter)
 */
const ParamAnimaModelSelect = () => {
  const animaLLLiteModel = useAppSelector(selectAnimaLLLiteModel);

  return (
    <>
      <ParamAnimaVaeModelSelect />
      <ParamAnimaQwen3EncoderModelSelect />
      <ParamAnimaLLLiteModelSelect />
      {animaLLLiteModel !== null && <ParamAnimaLLLiteWeight />}
    </>
  );
};

export default memo(ParamAnimaModelSelect);
