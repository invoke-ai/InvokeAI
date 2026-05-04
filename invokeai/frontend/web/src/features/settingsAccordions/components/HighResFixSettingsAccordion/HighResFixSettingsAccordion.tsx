import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  ButtonGroup,
  Combobox,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormControlGroup,
  FormLabel,
  StandaloneAccordion,
  Switch,
  Tooltip,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  selectBase,
  selectHrfEnabled,
  selectHrfFinalDimensions,
  selectHrfLatentInterpolationMode,
  selectHrfMethod,
  selectHrfScale,
  selectHrfStrength,
  selectHrfStructure,
  selectHrfTileControlNetModel,
  selectHrfTileOverlap,
  selectHrfTileSize,
  selectHrfUpscaleModel,
  selectIsRefinerModelSelected,
  selectModelSupportsHrf,
  setHrfEnabled,
  setHrfLatentInterpolationMode,
  setHrfMethod,
  setHrfScale,
  setHrfStrength,
  setHrfStructure,
  setHrfTileControlNetModel,
  setHrfTileOverlap,
  setHrfTileSize,
  setHrfUpscaleModel,
} from 'features/controlLayers/store/paramsSlice';
import { zHrfLatentInterpolationMode, zHrfMethod } from 'features/controlLayers/store/types';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import { useControlNetModels, useSpandrelImageToImageModels } from 'services/api/hooks/modelsByType';
import {
  type ControlNetModelConfig,
  isControlNetModelConfig,
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

const STRUCTURE_CONSTRAINTS = {
  initial: 0,
  sliderMin: -10,
  sliderMax: 10,
  numberInputMin: -10,
  numberInputMax: 10,
  coarseStep: 1,
  fineStep: 1,
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

const selectBadges = createMemoizedSelector(
  [
    selectHrfEnabled,
    selectHrfMethod,
    selectHrfScale,
    selectHrfStrength,
    selectHrfFinalDimensions,
    selectHrfUpscaleModel,
  ],
  (enabled, method, scale, strength, finalDimensions, upscaleModel) => {
    if (!enabled) {
      return EMPTY_ARRAY;
    }

    const methodBadge = method === 'upscale_model' ? 'Model' : 'Latent';
    const badges = [
      methodBadge,
      `${scale}x`,
      `${Math.round(strength * 100)}%`,
      `${finalDimensions.width}x${finalDimensions.height}`,
    ];

    if (method === 'upscale_model' && upscaleModel) {
      badges.push(upscaleModel.name);
    }

    return badges;
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
    <FormControl w="min-content">
      <InformationalPopover feature="paramHrf">
        <FormLabel m={0}>{t('hrf.enableHrf')}</FormLabel>
      </InformationalPopover>
      <Switch isChecked={enabled} onChange={onChange} />
    </FormControl>
  );
});

ParamHrfEnabled.displayName = 'ParamHrfEnabled';

const ParamHrfMethod = memo(() => {
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
      <ButtonGroup size="sm" variant="outline">
        <Button colorScheme={method === 'latent' ? 'invokeBlue' : undefined} onClick={onClickLatent}>
          {t('hrf.latent')}
        </Button>
        <Button colorScheme={method === 'upscale_model' ? 'invokeBlue' : undefined} onClick={onClickUpscaleModel}>
          {t('hrf.upscaleModelMethod')}
        </Button>
      </ButtonGroup>
    </FormControl>
  );
});

ParamHrfMethod.displayName = 'ParamHrfMethod';

const ParamHrfScale = memo(() => {
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
      />
      <CompositeNumberInput
        value={scale}
        defaultValue={SCALE_CONSTRAINTS.initial}
        min={SCALE_CONSTRAINTS.numberInputMin}
        max={SCALE_CONSTRAINTS.numberInputMax}
        step={SCALE_CONSTRAINTS.coarseStep}
        fineStep={SCALE_CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
});

ParamHrfScale.displayName = 'ParamHrfScale';

const ParamHrfStrength = memo(() => {
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
      />
      <CompositeNumberInput
        value={strength}
        defaultValue={STRENGTH_CONSTRAINTS.initial}
        min={STRENGTH_CONSTRAINTS.numberInputMin}
        max={STRENGTH_CONSTRAINTS.numberInputMax}
        step={STRENGTH_CONSTRAINTS.coarseStep}
        fineStep={STRENGTH_CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
});

ParamHrfStrength.displayName = 'ParamHrfStrength';

const ParamHrfLatentInterpolationMode = memo(() => {
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
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
});

ParamHrfLatentInterpolationMode.displayName = 'ParamHrfLatentInterpolationMode';

const ParamHrfUpscaleModel = memo(() => {
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
    <FormControl orientation="vertical">
      <InformationalPopover feature="upscaleModel">
        <FormLabel>{t('upscaling.upscaleModel')}</FormLabel>
      </InformationalPopover>
      <Flex w="full" alignItems="center" gap={2}>
        <Tooltip label={tooltipLabel}>
          <Box w="full">
            <Combobox
              value={value}
              placeholder={placeholder}
              options={options}
              onChange={onChange}
              noOptionsMessage={noOptionsMessage}
              isDisabled={options.length === 0}
            />
          </Box>
        </Tooltip>
      </Flex>
    </FormControl>
  );
});

ParamHrfUpscaleModel.displayName = 'ParamHrfUpscaleModel';

const ParamHrfTileControlNetModel = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const tileControlNetModel = useAppSelector(selectHrfTileControlNetModelConfig);
  const currentBaseModel = useAppSelector(selectBase);
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

  return (
    <FormControl
      isDisabled={!filteredModelConfigs.length}
      isInvalid={!filteredModelConfigs.length}
      minW={0}
      flexGrow={1}
      gap={2}
    >
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
        isDisabled={isLoading || !filteredModelConfigs.length}
        isInvalid={!filteredModelConfigs.length}
      />
    </FormControl>
  );
});

ParamHrfTileControlNetModel.displayName = 'ParamHrfTileControlNetModel';

const ParamHrfStructure = memo(() => {
  const dispatch = useAppDispatch();
  const structure = useAppSelector(selectHrfStructure);
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfStructure(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="structure">
        <FormLabel>{t('upscaling.structure')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={structure}
        defaultValue={STRUCTURE_CONSTRAINTS.initial}
        min={STRUCTURE_CONSTRAINTS.sliderMin}
        max={STRUCTURE_CONSTRAINTS.sliderMax}
        step={STRUCTURE_CONSTRAINTS.coarseStep}
        fineStep={STRUCTURE_CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={[STRUCTURE_CONSTRAINTS.sliderMin, STRUCTURE_CONSTRAINTS.initial, STRUCTURE_CONSTRAINTS.sliderMax]}
      />
      <CompositeNumberInput
        value={structure}
        defaultValue={STRUCTURE_CONSTRAINTS.initial}
        min={STRUCTURE_CONSTRAINTS.numberInputMin}
        max={STRUCTURE_CONSTRAINTS.numberInputMax}
        step={STRUCTURE_CONSTRAINTS.coarseStep}
        fineStep={STRUCTURE_CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
});

ParamHrfStructure.displayName = 'ParamHrfStructure';

const ParamHrfTileSize = memo(() => {
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
      />
      <CompositeNumberInput
        value={tileSize}
        defaultValue={TILE_SIZE_CONSTRAINTS.initial}
        min={TILE_SIZE_CONSTRAINTS.numberInputMin}
        max={TILE_SIZE_CONSTRAINTS.numberInputMax}
        step={TILE_SIZE_CONSTRAINTS.coarseStep}
        fineStep={TILE_SIZE_CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
});

ParamHrfTileSize.displayName = 'ParamHrfTileSize';

const ParamHrfTileOverlap = memo(() => {
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
      />
      <CompositeNumberInput
        value={tileOverlap}
        defaultValue={TILE_OVERLAP_CONSTRAINTS.initial}
        min={TILE_OVERLAP_CONSTRAINTS.numberInputMin}
        max={TILE_OVERLAP_CONSTRAINTS.numberInputMax}
        step={TILE_OVERLAP_CONSTRAINTS.coarseStep}
        fineStep={TILE_OVERLAP_CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
});

ParamHrfTileOverlap.displayName = 'ParamHrfTileOverlap';

export const HighResFixSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const badges = useAppSelector(selectBadges);
  const enabled = useAppSelector(selectHrfEnabled);
  const method = useAppSelector(selectHrfMethod);
  const modelSupportsHrf = useAppSelector(selectModelSupportsHrf);
  const isRefinerModelSelected = useAppSelector(selectIsRefinerModelSelected);
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: 'high-res-fix-settings-generate-tab',
    defaultIsOpen: false,
  });

  const parsedMethod = zHrfMethod.parse(method);

  if (!modelSupportsHrf || isRefinerModelSelected) {
    return null;
  }

  return (
    <StandaloneAccordion label={t('hrf.hrf')} badges={badges} isOpen={isOpen} onToggle={onToggle}>
      <Flex px={4} pt={4} pb={4} w="full" h="full" flexDir="column" gap={4}>
        <ParamHrfEnabled />
        {enabled && (
          <FormControlGroup>
            <ParamHrfMethod />
            <ParamHrfScale />
            <ParamHrfStrength />
            {parsedMethod === 'latent' ? (
              <ParamHrfLatentInterpolationMode />
            ) : (
              <>
                <ParamHrfUpscaleModel />
                <ParamHrfTileControlNetModel />
                <ParamHrfStructure />
                <ParamHrfTileSize />
                <ParamHrfTileOverlap />
              </>
            )}
          </FormControlGroup>
        )}
      </Flex>
    </StandaloneAccordion>
  );
});

HighResFixSettingsAccordion.displayName = 'HighResFixSettingsAccordion';
