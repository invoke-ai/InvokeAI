import type { ComboboxOnChange, FormLabelProps } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  ButtonGroup,
  Card,
  CardBody,
  CardHeader,
  Combobox,
  CompositeNumberInput,
  CompositeSlider,
  Expander,
  Flex,
  FormControl,
  FormControlGroup,
  FormLabel,
  IconButton,
  StandaloneAccordion,
  Switch,
  Text,
  Tooltip,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { DEFAULT_LORA_WEIGHT_CONFIG } from 'features/controlLayers/store/lorasSlice';
import {
  buildSelectHrfLoRA,
  hrfLoraAdded,
  hrfLoraDeleted,
  hrfLoraIsEnabledChanged,
  hrfLoraWeightChanged,
  selectBase,
  selectHrfEnabled,
  selectHrfFinalDimensions,
  selectHrfLatentInterpolationMode,
  selectHrfLoraMode,
  selectHrfLoras,
  selectHrfMethod,
  selectHrfModel,
  selectHrfScale,
  selectHrfSteps,
  selectHrfStrength,
  selectHrfTileControlEnd,
  selectHrfTileControlNetModel,
  selectHrfTileControlWeight,
  selectHrfTileOverlap,
  selectHrfTileSize,
  selectHrfUpscaleModel,
  selectIsRefinerModelSelected,
  selectModelSupportsHrf,
  selectSteps,
  setHrfEnabled,
  setHrfLatentInterpolationMode,
  setHrfLoraMode,
  setHrfMethod,
  setHrfModel,
  setHrfScale,
  setHrfSteps,
  setHrfStrength,
  setHrfTileControlEnd,
  setHrfTileControlNetModel,
  setHrfTileControlWeight,
  setHrfTileOverlap,
  setHrfTileSize,
  setHrfUpscaleModel,
} from 'features/controlLayers/store/paramsSlice';
import type { LoRA } from 'features/controlLayers/store/types';
import { zHrfLatentInterpolationMode, zHrfMethod } from 'features/controlLayers/store/types';
import { CONSTRAINTS as STEPS_CONSTRAINTS } from 'features/parameters/components/Core/ParamSteps';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import {
  modelConfigsAdapterSelectors,
  selectModelConfigsQuery,
  useGetModelConfigQuery,
} from 'services/api/endpoints/models';
import {
  useControlNetModels,
  useLoRAModels,
  useMainModels,
  useSpandrelImageToImageModels,
} from 'services/api/hooks/modelsByType';
import {
  type ControlNetModelConfig,
  isControlNetModelConfig,
  isExternalApiModelConfig,
  isMainOrExternalModelConfig,
  type LoRAModelConfig,
  type MainOrExternalModelConfig,
  type SpandrelImageToImageModelConfig,
} from 'services/api/types';

const SCALE_CONSTRAINTS = {
  initial: 2,
  sliderMin: 1,
  sliderMax: 4,
  numberInputMin: 1,
  numberInputMax: 8,
  coarseStep: 0.05,
  fineStep: 0.01,
};

const STRENGTH_CONSTRAINTS = {
  initial: 0.45,
  sliderMin: 0,
  sliderMax: 1,
  numberInputMin: 0,
  numberInputMax: 1,
  coarseStep: 0.01,
  fineStep: 0.01,
};

const TILE_CONTROL_WEIGHT_CONSTRAINTS = {
  initial: 0.625,
  sliderMin: 0,
  sliderMax: 1.5,
  numberInputMin: 0,
  numberInputMax: 2,
  coarseStep: 0.025,
  fineStep: 0.005,
};

const TILE_CONTROL_END_CONSTRAINTS = {
  initial: 0.2,
  sliderMin: 0,
  sliderMax: 1,
  numberInputMin: 0,
  numberInputMax: 1,
  coarseStep: 0.01,
  fineStep: 0.01,
};

const TILE_SIZE_CONSTRAINTS = {
  initial: 1024,
  sliderMin: 512,
  sliderMax: 1536,
  numberInputMin: 512,
  numberInputMax: 1536,
  coarseStep: 64,
  fineStep: 64,
};

const TILE_OVERLAP_CONSTRAINTS = {
  initial: 128,
  sliderMin: 32,
  sliderMax: 256,
  numberInputMin: 16,
  numberInputMax: 512,
  coarseStep: 16,
  fineStep: 8,
};

const formLabelProps: FormLabelProps = {
  m: 0,
  w: '10.5rem',
  minW: '10.5rem',
  maxW: '10.5rem',
  flexShrink: 0,
  whiteSpace: 'normal',
  lineHeight: 1.2,
  overflowWrap: 'break-word',
};

const formControlProps = {
  alignItems: 'center',
  gap: 3,
  minW: 0,
  w: 'full',
};

type DisabledProps = {
  isDisabled?: boolean;
};

const selectHrfTileControlNetModelConfig = createSelector(
  selectModelConfigsQuery,
  selectHrfTileControlNetModel,
  (modelConfigs, modelIdentifierField) => {
    if (!modelConfigs.data || !modelIdentifierField) {
      return null;
    }
    const modelConfig = modelConfigsAdapterSelectors.selectById(modelConfigs.data, modelIdentifierField.key);
    if (!modelConfig || !isControlNetModelConfig(modelConfig)) {
      return null;
    }
    return modelConfig;
  }
);

const selectHrfModelConfig = createSelector(selectModelConfigsQuery, selectHrfModel, (modelConfigs, model) => {
  if (!modelConfigs.data || !model) {
    return null;
  }
  const modelConfig = modelConfigsAdapterSelectors.selectById(modelConfigs.data, model.key);
  if (!modelConfig || !isMainOrExternalModelConfig(modelConfig) || isExternalApiModelConfig(modelConfig)) {
    return null;
  }
  return modelConfig;
});

const selectHrfLoRAIds = createMemoizedSelector(selectHrfLoras, (loras) => loras.map(({ id }) => id));
const selectHrfLoRAModelKeys = createMemoizedSelector(selectHrfLoras, (loras) => loras.map(({ model }) => model.key));

const selectBadges = createMemoizedSelector(
  [selectHrfEnabled, selectHrfMethod, selectHrfScale, selectHrfStrength, selectHrfFinalDimensions],
  (enabled, method, scale, strength, finalDimensions) => {
    if (!enabled) {
      return EMPTY_ARRAY;
    }

    const methodBadge = method === 'upscale_model' ? 'Model' : 'Latent';
    return [
      methodBadge,
      `${scale}x`,
      `${Math.round(strength * 100)}%`,
      `${finalDimensions.width}x${finalDimensions.height}`,
    ];
  }
);

const ParamHrfEnabled = memo(() => {
  const dispatch = useAppDispatch();
  const enabled = useAppSelector(selectHrfEnabled);
  const { t } = useTranslation();

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      dispatch(setHrfEnabled(event.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramHrf">
        <FormLabel m={0}>{t('hrf.enableHrf')}</FormLabel>
      </InformationalPopover>
      <Switch aria-label={t('hrf.enableUpscale')} isChecked={enabled} onChange={onChange} />
    </FormControl>
  );
});

ParamHrfEnabled.displayName = 'ParamHrfEnabled';

const ParamHrfMethod = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const method = useAppSelector(selectHrfMethod);
  const { t } = useTranslation();

  const onClickLatent = useCallback(() => {
    dispatch(setHrfMethod('latent'));
  }, [dispatch]);

  const onClickUpscaleModel = useCallback(() => {
    dispatch(setHrfMethod('upscale_model'));
  }, [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="paramUpscaleMethod">
        <FormLabel>{t('hrf.upscaleMethod')}</FormLabel>
      </InformationalPopover>
      <ButtonGroup size="sm" variant="outline" w="full">
        <Button
          flex={1}
          minW={0}
          colorScheme={method === 'latent' ? 'invokeBlue' : undefined}
          onClick={onClickLatent}
          isDisabled={isDisabled}
        >
          {t('hrf.latent')}
        </Button>
        <Button
          flex={1}
          minW={0}
          colorScheme={method === 'upscale_model' ? 'invokeBlue' : undefined}
          onClick={onClickUpscaleModel}
          isDisabled={isDisabled}
        >
          {t('hrf.upscaleModelMethod')}
        </Button>
      </ButtonGroup>
    </FormControl>
  );
});

ParamHrfMethod.displayName = 'ParamHrfMethod';

const ParamHrfScale = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const scale = useAppSelector(selectHrfScale);
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfScale(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="scale">
        <FormLabel>{t('hrf.scale')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={scale}
        defaultValue={SCALE_CONSTRAINTS.initial}
        min={SCALE_CONSTRAINTS.sliderMin}
        max={SCALE_CONSTRAINTS.sliderMax}
        step={SCALE_CONSTRAINTS.coarseStep}
        fineStep={SCALE_CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={[SCALE_CONSTRAINTS.sliderMin, SCALE_CONSTRAINTS.initial, SCALE_CONSTRAINTS.sliderMax]}
        isDisabled={isDisabled}
      />
      <CompositeNumberInput
        value={scale}
        defaultValue={SCALE_CONSTRAINTS.initial}
        min={SCALE_CONSTRAINTS.numberInputMin}
        max={SCALE_CONSTRAINTS.numberInputMax}
        step={SCALE_CONSTRAINTS.coarseStep}
        fineStep={SCALE_CONSTRAINTS.fineStep}
        onChange={onChange}
        isDisabled={isDisabled}
      />
    </FormControl>
  );
});

ParamHrfScale.displayName = 'ParamHrfScale';

const ParamHrfStrength = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const strength = useAppSelector(selectHrfStrength);
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfStrength(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramDenoisingStrength">
        <FormLabel>{t('hrf.strength')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={strength}
        defaultValue={STRENGTH_CONSTRAINTS.initial}
        min={STRENGTH_CONSTRAINTS.sliderMin}
        max={STRENGTH_CONSTRAINTS.sliderMax}
        step={STRENGTH_CONSTRAINTS.coarseStep}
        fineStep={STRENGTH_CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={[STRENGTH_CONSTRAINTS.sliderMin, STRENGTH_CONSTRAINTS.initial, STRENGTH_CONSTRAINTS.sliderMax]}
        isDisabled={isDisabled}
      />
      <CompositeNumberInput
        value={strength}
        defaultValue={STRENGTH_CONSTRAINTS.initial}
        min={STRENGTH_CONSTRAINTS.numberInputMin}
        max={STRENGTH_CONSTRAINTS.numberInputMax}
        step={STRENGTH_CONSTRAINTS.coarseStep}
        fineStep={STRENGTH_CONSTRAINTS.fineStep}
        onChange={onChange}
        isDisabled={isDisabled}
      />
    </FormControl>
  );
});

ParamHrfStrength.displayName = 'ParamHrfStrength';

const ParamHrfLatentInterpolationMode = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectHrfLatentInterpolationMode);
  const { t } = useTranslation();

  const options = useMemo(
    () => [
      { label: t('hrf.bilinear'), value: 'bilinear' },
      { label: t('hrf.bicubic'), value: 'bicubic' },
      { label: t('hrf.nearest'), value: 'nearest' },
      { label: t('hrf.nearestExact'), value: 'nearest-exact' },
      { label: t('hrf.area'), value: 'area' },
    ],
    [t]
  );

  const value = useMemo(() => options.find((o) => o.value === mode), [mode, options]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      const result = zHrfLatentInterpolationMode.safeParse(v?.value);
      if (!result.success) {
        return;
      }
      dispatch(setHrfLatentInterpolationMode(result.data));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramUpscaleMethod">
        <FormLabel>{t('hrf.latentInterpolationMode')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={onChange} isDisabled={isDisabled} />
    </FormControl>
  );
});

ParamHrfLatentInterpolationMode.displayName = 'ParamHrfLatentInterpolationMode';

const ParamHrfUpscaleModel = memo(({ isDisabled = false }: DisabledProps) => {
  const { t } = useTranslation();
  const [modelConfigs, { isLoading }] = useSpandrelImageToImageModels();
  const model = useAppSelector(selectHrfUpscaleModel);
  const dispatch = useAppDispatch();

  const tooltipLabel = useMemo(() => {
    if (!modelConfigs.length || !model) {
      return;
    }
    return modelConfigs.find((m) => m.key === model.key)?.description;
  }, [modelConfigs, model]);

  const _onChange = useCallback(
    (v: SpandrelImageToImageModelConfig | null) => {
      dispatch(setHrfUpscaleModel(v));
    },
    [dispatch]
  );

  const { options, value, onChange, placeholder, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: model,
    isLoading,
  });

  return (
    <FormControl isDisabled={isDisabled}>
      <InformationalPopover feature="upscaleModel">
        <FormLabel>{t('upscaling.upscaleModel')}</FormLabel>
      </InformationalPopover>
      <Flex w="full" alignItems="center" gap={2}>
        <Tooltip label={tooltipLabel}>
          <Box w="full" minW={0}>
            <Combobox
              value={value}
              placeholder={placeholder}
              options={options}
              onChange={onChange}
              noOptionsMessage={noOptionsMessage}
              isDisabled={isDisabled || options.length === 0}
            />
          </Box>
        </Tooltip>
      </Flex>
    </FormControl>
  );
});

ParamHrfUpscaleModel.displayName = 'ParamHrfUpscaleModel';

const ParamHrfTileControlNetModel = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const tileControlNetModel = useAppSelector(selectHrfTileControlNetModelConfig);
  const generateBaseModel = useAppSelector(selectBase);
  const hrfModel = useAppSelector(selectHrfModel);
  const currentBaseModel = hrfModel?.base ?? generateBaseModel;
  const [modelConfigs, { isLoading }] = useControlNetModels();

  const onChange = useCallback(
    (controlNetModel: ControlNetModelConfig) => {
      dispatch(setHrfTileControlNetModel(controlNetModel));
    },
    [dispatch]
  );

  const filteredModelConfigs = useMemo(() => {
    if (!currentBaseModel || !['sd-1', 'sdxl'].includes(currentBaseModel)) {
      return [];
    }
    return modelConfigs.filter((model) => {
      const isCompatible = model.base === currentBaseModel;
      const isTileOrMultiModel =
        model.name.toLowerCase().includes('tile') || model.name.toLowerCase().includes('union');
      return isCompatible && isTileOrMultiModel;
    });
  }, [modelConfigs, currentBaseModel]);

  const getIsOptionDisabled = useCallback(
    (model: ControlNetModelConfig): boolean => {
      return currentBaseModel !== model.base;
    },
    [currentBaseModel]
  );
  const isMissingModel = !filteredModelConfigs.length;
  const isInvalid = !isDisabled && isMissingModel;

  return (
    <FormControl isDisabled={isDisabled || isMissingModel} isInvalid={isInvalid} minW={0} flexGrow={1} gap={2}>
      <InformationalPopover feature="controlNet">
        <FormLabel m={0}>{t('upscaling.tileControl')}</FormLabel>
      </InformationalPopover>
      <ModelPicker
        pickerId="hrf-tile-controlnet-model"
        modelConfigs={filteredModelConfigs}
        selectedModelConfig={tileControlNetModel ?? undefined}
        onChange={onChange}
        getIsOptionDisabled={getIsOptionDisabled}
        placeholder={t('common.placeholderSelectAModel')}
        noOptionsText={t('upscaling.missingTileControlNetModel')}
        isDisabled={isDisabled || isLoading || isMissingModel}
        isInvalid={isInvalid}
      />
    </FormControl>
  );
});

ParamHrfTileControlNetModel.displayName = 'ParamHrfTileControlNetModel';

const ParamHrfTileControlWeight = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const tileControlWeight = useAppSelector(selectHrfTileControlWeight);
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfTileControlWeight(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="controlNet">
        <FormLabel>{t('hrf.tileControlWeight')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={tileControlWeight}
        defaultValue={TILE_CONTROL_WEIGHT_CONSTRAINTS.initial}
        min={TILE_CONTROL_WEIGHT_CONSTRAINTS.sliderMin}
        max={TILE_CONTROL_WEIGHT_CONSTRAINTS.sliderMax}
        step={TILE_CONTROL_WEIGHT_CONSTRAINTS.coarseStep}
        fineStep={TILE_CONTROL_WEIGHT_CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={[
          TILE_CONTROL_WEIGHT_CONSTRAINTS.sliderMin,
          TILE_CONTROL_WEIGHT_CONSTRAINTS.initial,
          TILE_CONTROL_WEIGHT_CONSTRAINTS.sliderMax,
        ]}
        isDisabled={isDisabled}
      />
      <CompositeNumberInput
        value={tileControlWeight}
        defaultValue={TILE_CONTROL_WEIGHT_CONSTRAINTS.initial}
        min={TILE_CONTROL_WEIGHT_CONSTRAINTS.numberInputMin}
        max={TILE_CONTROL_WEIGHT_CONSTRAINTS.numberInputMax}
        step={TILE_CONTROL_WEIGHT_CONSTRAINTS.coarseStep}
        fineStep={TILE_CONTROL_WEIGHT_CONSTRAINTS.fineStep}
        onChange={onChange}
        isDisabled={isDisabled}
      />
    </FormControl>
  );
});

ParamHrfTileControlWeight.displayName = 'ParamHrfTileControlWeight';

const ParamHrfTileControlEnd = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const tileControlEnd = useAppSelector(selectHrfTileControlEnd);
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfTileControlEnd(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramDenoisingStrength">
        <FormLabel>{t('hrf.tileControlEnd')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={tileControlEnd}
        defaultValue={TILE_CONTROL_END_CONSTRAINTS.initial}
        min={TILE_CONTROL_END_CONSTRAINTS.sliderMin}
        max={TILE_CONTROL_END_CONSTRAINTS.sliderMax}
        step={TILE_CONTROL_END_CONSTRAINTS.coarseStep}
        fineStep={TILE_CONTROL_END_CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={[
          TILE_CONTROL_END_CONSTRAINTS.sliderMin,
          TILE_CONTROL_END_CONSTRAINTS.initial,
          TILE_CONTROL_END_CONSTRAINTS.sliderMax,
        ]}
        isDisabled={isDisabled}
      />
      <CompositeNumberInput
        value={tileControlEnd}
        defaultValue={TILE_CONTROL_END_CONSTRAINTS.initial}
        min={TILE_CONTROL_END_CONSTRAINTS.numberInputMin}
        max={TILE_CONTROL_END_CONSTRAINTS.numberInputMax}
        step={TILE_CONTROL_END_CONSTRAINTS.coarseStep}
        fineStep={TILE_CONTROL_END_CONSTRAINTS.fineStep}
        onChange={onChange}
        isDisabled={isDisabled}
      />
    </FormControl>
  );
});

ParamHrfTileControlEnd.displayName = 'ParamHrfTileControlEnd';

const ParamHrfTileSize = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const tileSize = useAppSelector(selectHrfTileSize);
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfTileSize(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="tileSize">
        <FormLabel>{t('upscaling.tileSize')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={tileSize}
        defaultValue={TILE_SIZE_CONSTRAINTS.initial}
        min={TILE_SIZE_CONSTRAINTS.sliderMin}
        max={TILE_SIZE_CONSTRAINTS.sliderMax}
        step={TILE_SIZE_CONSTRAINTS.coarseStep}
        fineStep={TILE_SIZE_CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={[TILE_SIZE_CONSTRAINTS.sliderMin, TILE_SIZE_CONSTRAINTS.initial, TILE_SIZE_CONSTRAINTS.sliderMax]}
        isDisabled={isDisabled}
      />
      <CompositeNumberInput
        value={tileSize}
        defaultValue={TILE_SIZE_CONSTRAINTS.initial}
        min={TILE_SIZE_CONSTRAINTS.numberInputMin}
        max={TILE_SIZE_CONSTRAINTS.numberInputMax}
        step={TILE_SIZE_CONSTRAINTS.coarseStep}
        fineStep={TILE_SIZE_CONSTRAINTS.fineStep}
        onChange={onChange}
        isDisabled={isDisabled}
      />
    </FormControl>
  );
});

ParamHrfTileSize.displayName = 'ParamHrfTileSize';

const ParamHrfTileOverlap = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const tileOverlap = useAppSelector(selectHrfTileOverlap);
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfTileOverlap(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="tileOverlap">
        <FormLabel>{t('upscaling.tileOverlap')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={tileOverlap}
        defaultValue={TILE_OVERLAP_CONSTRAINTS.initial}
        min={TILE_OVERLAP_CONSTRAINTS.sliderMin}
        max={TILE_OVERLAP_CONSTRAINTS.sliderMax}
        step={TILE_OVERLAP_CONSTRAINTS.coarseStep}
        fineStep={TILE_OVERLAP_CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={[
          TILE_OVERLAP_CONSTRAINTS.sliderMin,
          TILE_OVERLAP_CONSTRAINTS.initial,
          TILE_OVERLAP_CONSTRAINTS.sliderMax,
        ]}
        isDisabled={isDisabled}
      />
      <CompositeNumberInput
        value={tileOverlap}
        defaultValue={TILE_OVERLAP_CONSTRAINTS.initial}
        min={TILE_OVERLAP_CONSTRAINTS.numberInputMin}
        max={TILE_OVERLAP_CONSTRAINTS.numberInputMax}
        step={TILE_OVERLAP_CONSTRAINTS.coarseStep}
        fineStep={TILE_OVERLAP_CONSTRAINTS.fineStep}
        onChange={onChange}
        isDisabled={isDisabled}
      />
    </FormControl>
  );
});

ParamHrfTileOverlap.displayName = 'ParamHrfTileOverlap';

const ParamHrfSteps = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const hrfSteps = useAppSelector(selectHrfSteps);
  const generateSteps = useAppSelector(selectSteps);
  const { t } = useTranslation();

  const onToggle = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      dispatch(setHrfSteps(event.target.checked ? generateSteps : null));
    },
    [dispatch, generateSteps]
  );

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfSteps(v));
    },
    [dispatch]
  );

  const isCustom = hrfSteps !== null;

  return (
    <FormControl>
      <InformationalPopover feature="paramSteps">
        <FormLabel>{t('hrf.steps')}</FormLabel>
      </InformationalPopover>
      <Flex alignItems="center" minW={0} flex={1}>
        <Switch size="sm" isChecked={isCustom} onChange={onToggle} flexShrink={0} isDisabled={isDisabled} />
      </Flex>
      <CompositeNumberInput
        value={hrfSteps ?? generateSteps}
        defaultValue={STEPS_CONSTRAINTS.initial}
        min={STEPS_CONSTRAINTS.numberInputMin}
        max={STEPS_CONSTRAINTS.numberInputMax}
        step={STEPS_CONSTRAINTS.coarseStep}
        fineStep={STEPS_CONSTRAINTS.fineStep}
        onChange={onChange}
        isDisabled={isDisabled || !isCustom}
        w={24}
        flexShrink={0}
      />
    </FormControl>
  );
});

ParamHrfSteps.displayName = 'ParamHrfSteps';

const ParamHrfModel = memo(({ isDisabled = false }: DisabledProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectedModelConfig = useAppSelector(selectHrfModelConfig);
  const currentBaseModel = useAppSelector(selectBase);

  const filter = useCallback(
    (model: MainOrExternalModelConfig) => {
      return (
        !isExternalApiModelConfig(model) && model.base === currentBaseModel && ['sd-1', 'sdxl'].includes(model.base)
      );
    },
    [currentBaseModel]
  );
  const [modelConfigs, { isLoading }] = useMainModels(filter);

  const onChange = useCallback(
    (model: MainOrExternalModelConfig | null) => {
      dispatch(setHrfModel(model && !isExternalApiModelConfig(model) ? model : null));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={isDisabled}>
      <InformationalPopover feature="paramModel">
        <FormLabel>{t('hrf.model')}</FormLabel>
      </InformationalPopover>
      <ModelPicker
        pickerId="hrf-model"
        modelConfigs={modelConfigs}
        selectedModelConfig={selectedModelConfig ?? undefined}
        onChange={onChange}
        grouped
        allowEmpty
        placeholder={t('hrf.reuseGenerateModel')}
        noOptionsText={currentBaseModel ? t('hrf.noCompatibleModels') : t('models.selectModel')}
        isDisabled={isDisabled || isLoading || !modelConfigs.length}
      />
    </FormControl>
  );
});

ParamHrfModel.displayName = 'ParamHrfModel';

const ParamHrfLoraMode = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectHrfLoraMode);
  const { t } = useTranslation();

  const onClickReuseGenerate = useCallback(() => {
    dispatch(setHrfLoraMode('reuse_generate'));
  }, [dispatch]);

  const onClickNone = useCallback(() => {
    dispatch(setHrfLoraMode('none'));
  }, [dispatch]);

  const onClickDedicated = useCallback(() => {
    dispatch(setHrfLoraMode('dedicated'));
  }, [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="lora">
        <FormLabel>{t('hrf.loraMode')}</FormLabel>
      </InformationalPopover>
      <ButtonGroup size="sm" variant="outline" w="full">
        <Button
          flex={1}
          minW={0}
          colorScheme={mode === 'reuse_generate' ? 'invokeBlue' : undefined}
          onClick={onClickReuseGenerate}
          isDisabled={isDisabled}
        >
          {t('hrf.reuseGenerateLoras')}
        </Button>
        <Button
          flex={1}
          minW={0}
          colorScheme={mode === 'none' ? 'invokeBlue' : undefined}
          onClick={onClickNone}
          isDisabled={isDisabled}
        >
          {t('hrf.noLoras')}
        </Button>
        <Button
          flex={1}
          minW={0}
          colorScheme={mode === 'dedicated' ? 'invokeBlue' : undefined}
          onClick={onClickDedicated}
          isDisabled={isDisabled}
        >
          {t('hrf.dedicatedLoras')}
        </Button>
      </ButtonGroup>
    </FormControl>
  );
});

ParamHrfLoraMode.displayName = 'ParamHrfLoraMode';

const ParamHrfLoraSelect = memo(({ isDisabled = false }: DisabledProps) => {
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useLoRAModels();
  const { t } = useTranslation();
  const addedLoRAModelKeys = useAppSelector(selectHrfLoRAModelKeys);
  const currentBaseModel = useAppSelector(selectBase);
  const hrfModel = useAppSelector(selectHrfModel);
  const hrfBase = hrfModel?.base ?? currentBaseModel;

  const compatibleLoRAs = useMemo(() => {
    if (!hrfBase) {
      return EMPTY_ARRAY;
    }
    return modelConfigs.filter((model) => model.base === hrfBase);
  }, [hrfBase, modelConfigs]);

  const getIsDisabled = useCallback(
    (model: LoRAModelConfig): boolean => {
      return addedLoRAModelKeys.includes(model.key);
    },
    [addedLoRAModelKeys]
  );

  const onChange = useCallback(
    (model: LoRAModelConfig | null) => {
      if (!model) {
        return;
      }
      dispatch(hrfLoraAdded({ model }));
    },
    [dispatch]
  );

  const placeholder = useMemo(() => {
    if (isLoading) {
      return t('common.loading');
    }
    if (compatibleLoRAs.length === 0) {
      return hrfBase ? t('models.noCompatibleLoRAs') : t('models.selectModel');
    }
    return t('hrf.addDedicatedLora');
  }, [compatibleLoRAs.length, hrfBase, isLoading, t]);

  return (
    <FormControl gap={2} isDisabled={isDisabled}>
      <InformationalPopover feature="lora">
        <FormLabel>{t('hrf.dedicatedLoras')}</FormLabel>
      </InformationalPopover>
      <ModelPicker
        pickerId="hrf-lora-select"
        modelConfigs={compatibleLoRAs}
        onChange={onChange}
        grouped={false}
        selectedModelConfig={undefined}
        allowEmpty
        placeholder={placeholder}
        getIsOptionDisabled={getIsDisabled}
        noOptionsText={hrfBase ? t('models.noCompatibleLoRAs') : t('models.selectModel')}
        isDisabled={isDisabled}
      />
    </FormControl>
  );
});

ParamHrfLoraSelect.displayName = 'ParamHrfLoraSelect';

const HrfLoRAList = memo(({ isDisabled = false }: DisabledProps) => {
  const ids = useAppSelector(selectHrfLoRAIds);

  if (!ids.length) {
    return null;
  }

  return (
    <Flex flexWrap="wrap" gap={2}>
      {ids.map((id) => (
        <HrfLoRACard key={id} id={id} isDisabled={isDisabled} />
      ))}
    </Flex>
  );
});

HrfLoRAList.displayName = 'HrfLoRAList';

const HrfLoRACard = memo((props: { id: string } & DisabledProps) => {
  const selectLoRA = useMemo(() => buildSelectHrfLoRA(props.id), [props.id]);
  const lora = useAppSelector(selectLoRA);

  if (!lora) {
    return null;
  }
  return <HrfLoRAContent lora={lora} isDisabled={props.isDisabled} />;
});

HrfLoRACard.displayName = 'HrfLoRACard';

const HrfLoRAContent = memo(({ lora, isDisabled = false }: { lora: LoRA } & DisabledProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { data: loraConfig } = useGetModelConfigQuery(lora.model.key);
  const isWeightDisabled = isDisabled || !lora.isEnabled;

  const onChange = useCallback(
    (v: number) => {
      dispatch(hrfLoraWeightChanged({ id: lora.id, weight: v }));
    },
    [dispatch, lora.id]
  );

  const onToggle = useCallback(() => {
    dispatch(hrfLoraIsEnabledChanged({ id: lora.id, isEnabled: !lora.isEnabled }));
  }, [dispatch, lora.id, lora.isEnabled]);

  const onRemove = useCallback(() => {
    dispatch(hrfLoraDeleted({ id: lora.id }));
  }, [dispatch, lora.id]);

  return (
    <Card variant="lora">
      <CardHeader>
        <Flex alignItems="center" justifyContent="space-between" width="100%" gap={2}>
          <Text noOfLines={1} wordBreak="break-all" color={lora.isEnabled && !isDisabled ? 'base.200' : 'base.500'}>
            {loraConfig?.name ?? lora.model.key.substring(0, 8)}
          </Text>
          <Flex alignItems="center" gap={2}>
            <Switch size="sm" onChange={onToggle} isChecked={lora.isEnabled} isDisabled={isDisabled} />
            <IconButton
              aria-label={t('lora.removeLoRA')}
              variant="ghost"
              size="sm"
              onClick={onRemove}
              icon={<PiTrashSimpleBold />}
              isDisabled={isDisabled}
            />
          </Flex>
        </Flex>
      </CardHeader>
      <InformationalPopover feature="loraWeight">
        <CardBody>
          <CompositeSlider
            value={lora.weight}
            onChange={onChange}
            min={DEFAULT_LORA_WEIGHT_CONFIG.sliderMin}
            max={DEFAULT_LORA_WEIGHT_CONFIG.sliderMax}
            step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
            fineStep={DEFAULT_LORA_WEIGHT_CONFIG.fineStep}
            marks={[-1, 0, 1, 2]}
            defaultValue={DEFAULT_LORA_WEIGHT_CONFIG.initial}
            isDisabled={isWeightDisabled}
          />
          <CompositeNumberInput
            value={lora.weight}
            onChange={onChange}
            min={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
            max={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
            step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
            fineStep={DEFAULT_LORA_WEIGHT_CONFIG.fineStep}
            w={20}
            flexShrink={0}
            defaultValue={DEFAULT_LORA_WEIGHT_CONFIG.initial}
            isDisabled={isWeightDisabled}
          />
        </CardBody>
      </InformationalPopover>
    </Card>
  );
});

HrfLoRAContent.displayName = 'HrfLoRAContent';

export const HighResFixSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const badges = useAppSelector(selectBadges);
  const hrfEnabled = useAppSelector(selectHrfEnabled);
  const method = useAppSelector(selectHrfMethod);
  const modelSupportsHrf = useAppSelector(selectModelSupportsHrf);
  const isRefinerModelSelected = useAppSelector(selectIsRefinerModelSelected);
  const hrfLoraMode = useAppSelector(selectHrfLoraMode);
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: 'high-res-fix-settings-generate-tab',
    defaultIsOpen: false,
  });
  const { isOpen: isOpenExpander, onToggle: onToggleExpander } = useExpanderToggle({
    id: 'high-res-fix-settings-generate-tab-advanced',
    defaultIsOpen: false,
  });

  const parsedMethod = zHrfMethod.parse(method);
  const isDisabled = !hrfEnabled;

  if (!modelSupportsHrf || isRefinerModelSelected) {
    return null;
  }

  return (
    <StandaloneAccordion label={t('hrf.hrf')} badges={badges} isOpen={isOpen} onToggle={onToggle}>
      <Flex px={4} pt={4} pb={0} w="full" h="full" flexDir="column" gap={4}>
        <FormControlGroup formLabelProps={formLabelProps} formControlProps={formControlProps}>
          <ParamHrfEnabled />
        </FormControlGroup>
        <FormControlGroup formLabelProps={formLabelProps} formControlProps={formControlProps} isDisabled={isDisabled}>
          <ParamHrfMethod isDisabled={isDisabled} />
          <ParamHrfScale isDisabled={isDisabled} />
          <ParamHrfStrength isDisabled={isDisabled} />
          {parsedMethod === 'upscale_model' && (
            <>
              <ParamHrfUpscaleModel isDisabled={isDisabled} />
              <ParamHrfTileControlNetModel isDisabled={isDisabled} />
            </>
          )}
        </FormControlGroup>
      </Flex>
      <Expander label={t('accordions.advanced.options')} isOpen={isOpenExpander} onToggle={onToggleExpander}>
        <Flex gap={4} flexDir="column" px={4} pb={4}>
          <FormControlGroup formLabelProps={formLabelProps} formControlProps={formControlProps} isDisabled={isDisabled}>
            {parsedMethod === 'latent' && <ParamHrfLatentInterpolationMode isDisabled={isDisabled} />}
            {parsedMethod === 'upscale_model' && (
              <>
                <ParamHrfTileControlWeight isDisabled={isDisabled} />
                <ParamHrfTileControlEnd isDisabled={isDisabled} />
                <ParamHrfTileSize isDisabled={isDisabled} />
                <ParamHrfTileOverlap isDisabled={isDisabled} />
                <ParamHrfModel isDisabled={isDisabled} />
                <ParamHrfLoraMode isDisabled={isDisabled} />
                <ParamHrfSteps isDisabled={isDisabled} />
              </>
            )}
          </FormControlGroup>
          {parsedMethod === 'upscale_model' && hrfLoraMode === 'dedicated' && (
            <>
              <ParamHrfLoraSelect isDisabled={isDisabled} />
              <HrfLoRAList isDisabled={isDisabled} />
            </>
          )}
        </Flex>
      </Expander>
    </StandaloneAccordion>
  );
});

HighResFixSettingsAccordion.displayName = 'HighResFixSettingsAccordion';
