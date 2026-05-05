import type { ComboboxOnChange, ComboboxOption, FormLabelProps } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  ButtonGroup,
  Combobox,
  CompositeNumberInput,
  CompositeSlider,
  Expander,
  Flex,
  FormControl,
  FormControlGroup,
  FormLabel,
  Grid,
  IconButton,
  Input,
  StandaloneAccordion,
  Switch,
  Tooltip,
} from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectDetailerCfgScale,
  selectDetailerColorCorrectMode,
  selectDetailerCropPadding,
  selectDetailerDenoiseMaskExpand,
  selectDetailerDenoiseMaskFeather,
  selectDetailerDetectionThreshold,
  selectDetailerDetector,
  selectDetailerDinoModel,
  selectDetailerEnabled,
  selectDetailerFaceId,
  selectDetailerFaceSelection,
  selectDetailerMaskBlur,
  selectDetailerMaxProcessSize,
  selectDetailerMaxUpscale,
  selectDetailerMinConfidence,
  selectDetailerPadding,
  selectDetailerPasteMaskExpand,
  selectDetailerPasteMaskFeather,
  selectDetailerQuality,
  selectDetailerSamModel,
  selectDetailerSteps,
  selectDetailerStrength,
  selectDetailerTargetPrompt,
  selectDetailerTargetSize,
  selectModelSupportsFaceDetailer,
  setDetailerCfgScale,
  setDetailerColorCorrectMode,
  setDetailerCropPadding,
  setDetailerDenoiseMaskExpand,
  setDetailerDenoiseMaskFeather,
  setDetailerDetectionThreshold,
  setDetailerDetector,
  setDetailerDinoModel,
  setDetailerEnabled,
  setDetailerFaceId,
  setDetailerFaceSelection,
  setDetailerMaskBlur,
  setDetailerMaxProcessSize,
  setDetailerMaxUpscale,
  setDetailerMinConfidence,
  setDetailerPadding,
  setDetailerPasteMaskExpand,
  setDetailerPasteMaskFeather,
  setDetailerQuality,
  setDetailerSamModel,
  setDetailerSteps,
  setDetailerStrength,
  setDetailerTargetPrompt,
  setDetailerTargetSize,
} from 'features/controlLayers/store/paramsSlice';
import type {
  DetailerColorCorrectMode,
  DetailerDetector,
  DetailerDinoModel,
  DetailerFaceSelection,
  DetailerQuality,
  DetailerSamModel,
} from 'features/controlLayers/store/types';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import type { ChangeEvent, MouseEventHandler, ReactNode } from 'react';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiHandBold, PiHeadCircuitBold, PiPencilSimpleBold, PiPersonBold, PiSmileyBold } from 'react-icons/pi';

const formLabelProps: FormLabelProps = {
  minW: '7.5rem',
};

const compactFormLabelProps: FormLabelProps = {
  minW: '4.5rem',
};

const DETAILER_DETECTORS = ['grounding-dino-sam', 'mediapipe'] as const;
const DETAILER_QUALITIES = ['fast', 'balanced', 'high'] as const;
const DETAILER_FACE_SELECTIONS = ['highest_score', 'largest_area', 'index'] as const;
const DETAILER_DINO_MODELS = ['grounding-dino-tiny', 'grounding-dino-base'] as const;
const DETAILER_COLOR_CORRECT_MODES = ['off', 'YCbCr-Luma', 'YCbCr-Chroma', 'YCbCr', 'RGB'] as const;
const DETAILER_SAM_MODELS = [
  'segment-anything-2-small',
  'segment-anything-2-tiny',
  'segment-anything-2-base',
  'segment-anything-2-large',
  'segment-anything-base',
  'segment-anything-large',
  'segment-anything-huge',
] as const;

const DETAILER_TARGET_PRESETS = [
  { labelKey: 'parameters.faceDetailer.targetPresets.face', prompt: 'face', icon: PiSmileyBold },
  { labelKey: 'parameters.faceDetailer.targetPresets.head', prompt: 'head', icon: PiHeadCircuitBold },
  { labelKey: 'parameters.faceDetailer.targetPresets.hands', prompt: 'hands', icon: PiHandBold },
  { labelKey: 'parameters.faceDetailer.targetPresets.body', prompt: 'person', icon: PiPersonBold },
] as const;

const DETAILER_TARGET_PRESET_PROMPTS = new Set<string>(DETAILER_TARGET_PRESETS.map(({ prompt }) => prompt));
const DETAILER_TARGET_BUTTON_GRID_TEMPLATE = 'repeat(5, minmax(3.25rem, 3.5rem))';

const getTargetButtonStyleProps = (isSelected: boolean) => ({
  aspectRatio: 1,
  bg: 'base.800',
  borderColor: 'base.700',
  borderRadius: 'base',
  borderStyle: 'solid',
  borderWidth: 1,
  borderBottomColor: isSelected ? 'invokeBlue.400' : 'transparent',
  borderBottomWidth: 2,
  boxShadow: 'none',
  color: isSelected ? 'base.50' : 'base.300',
  h: 'auto',
  minH: 13,
  minW: 0,
  variant: 'ghost',
  w: 'full',
  _active: {
    bg: 'base.700',
  },
  _disabled: {
    cursor: 'not-allowed',
    opacity: 0.4,
  },
  _hover: {
    bg: 'base.700',
    borderBottomColor: isSelected ? 'invokeBlue.300' : 'transparent',
    color: 'base.50',
  },
});

const FACE_ID_CONSTRAINTS = {
  initial: 0,
  sliderMin: 0,
  sliderMax: 10,
  numberInputMin: 0,
  numberInputMax: 100,
  step: 1,
  fineStep: 1,
};

const CONFIDENCE_CONSTRAINTS = {
  initial: 0.5,
  sliderMin: 0,
  sliderMax: 1,
  numberInputMin: 0,
  numberInputMax: 1,
  step: 0.05,
  fineStep: 0.01,
};

const PADDING_CONSTRAINTS = {
  initial: 32,
  sliderMin: 0,
  sliderMax: 128,
  numberInputMin: 0,
  numberInputMax: 512,
  step: 8,
  fineStep: 1,
};

const CROP_PADDING_CONSTRAINTS = {
  initial: 64,
  sliderMin: 0,
  sliderMax: 256,
  numberInputMin: 0,
  numberInputMax: 512,
  step: 8,
  fineStep: 1,
};

const STRENGTH_CONSTRAINTS = {
  initial: 0.28,
  sliderMin: 0,
  sliderMax: 1,
  numberInputMin: 0,
  numberInputMax: 1,
  step: 0.05,
  fineStep: 0.01,
};

const STEPS_CONSTRAINTS = {
  initial: 14,
  sliderMin: 1,
  sliderMax: 50,
  numberInputMin: 1,
  numberInputMax: 500,
  step: 1,
  fineStep: 1,
};

const CFG_SCALE_CONSTRAINTS = {
  initial: 7.5,
  sliderMin: 1,
  sliderMax: 20,
  numberInputMin: 1,
  numberInputMax: 200,
  step: 0.5,
  fineStep: 0.1,
};

const MASK_BLUR_CONSTRAINTS = {
  initial: 8,
  sliderMin: 0,
  sliderMax: 64,
  numberInputMin: 0,
  numberInputMax: 256,
  step: 1,
  fineStep: 1,
};

const TARGET_SIZE_CONSTRAINTS = {
  initial: 768,
  sliderMin: 64,
  sliderMax: 1024,
  numberInputMin: 64,
  numberInputMax: 2048,
  step: 64,
  fineStep: 8,
};

const MAX_UPSCALE_CONSTRAINTS = {
  initial: 8,
  sliderMin: 1,
  sliderMax: 12,
  numberInputMin: 1,
  numberInputMax: 16,
  step: 0.5,
  fineStep: 0.1,
};

const MAX_PROCESS_SIZE_CONSTRAINTS = {
  initial: 768,
  sliderMin: 256,
  sliderMax: 1536,
  numberInputMin: 64,
  numberInputMax: 2048,
  step: 64,
  fineStep: 8,
};

const DENOISE_MASK_EXPAND_CONSTRAINTS = {
  initial: 8,
  sliderMin: 0,
  sliderMax: 64,
  numberInputMin: 0,
  numberInputMax: 256,
  step: 1,
  fineStep: 1,
};

const DENOISE_MASK_FEATHER_CONSTRAINTS = {
  initial: 8,
  sliderMin: 0,
  sliderMax: 96,
  numberInputMin: 0,
  numberInputMax: 256,
  step: 1,
  fineStep: 1,
};

const PASTE_MASK_EXPAND_CONSTRAINTS = {
  initial: 0,
  sliderMin: 0,
  sliderMax: 64,
  numberInputMin: 0,
  numberInputMax: 256,
  step: 1,
  fineStep: 1,
};

const PASTE_MASK_FEATHER_CONSTRAINTS = {
  initial: 8,
  sliderMin: 0,
  sliderMax: 96,
  numberInputMin: 0,
  numberInputMax: 256,
  step: 1,
  fineStep: 1,
};

const isInValues = <T extends string>(value: unknown, values: readonly T[]): value is T =>
  typeof value === 'string' && values.includes(value as T);

type FaceDetailerNumberParamProps = {
  label: string;
  value: number;
  defaultValue: number;
  sliderMin: number;
  sliderMax: number;
  numberInputMin: number;
  numberInputMax: number;
  step: number;
  fineStep: number;
  onChange: (value: number) => void;
  isDisabled?: boolean;
};

const FaceDetailerNumberParam = memo(
  ({
    label,
    value,
    defaultValue,
    sliderMin,
    sliderMax,
    numberInputMin,
    numberInputMax,
    step,
    fineStep,
    onChange,
    isDisabled,
  }: FaceDetailerNumberParamProps) => (
    <FormControl isDisabled={isDisabled}>
      <FormLabel>{label}</FormLabel>
      <CompositeSlider
        value={value}
        defaultValue={defaultValue}
        min={sliderMin}
        max={sliderMax}
        step={step}
        fineStep={fineStep}
        onChange={onChange}
      />
      <CompositeNumberInput
        value={value}
        defaultValue={defaultValue}
        min={numberInputMin}
        max={numberInputMax}
        step={step}
        fineStep={fineStep}
        onChange={onChange}
      />
    </FormControl>
  )
);

FaceDetailerNumberParam.displayName = 'FaceDetailerNumberParam';

type FaceDetailerSelectParamProps = {
  label: string;
  value: string;
  options: ComboboxOption[];
  onChange: ComboboxOnChange;
  isDisabled?: boolean;
};

const FaceDetailerSelectParam = memo(
  ({ label, value, options, onChange, isDisabled }: FaceDetailerSelectParamProps) => {
    const selected = useMemo(() => options.find((option) => option.value === value), [options, value]);

    return (
      <FormControl isDisabled={isDisabled}>
        <FormLabel>{label}</FormLabel>
        <Combobox value={selected} options={options} onChange={onChange} isSearchable={false} isClearable={false} />
      </FormControl>
    );
  }
);

FaceDetailerSelectParam.displayName = 'FaceDetailerSelectParam';

type FaceDetailerTargetPresetButtonsProps = {
  value: string;
  onChange: (value: string) => void;
  customInput: ReactNode;
  isDisabled?: boolean;
};

const FaceDetailerTargetPresetButtons = memo(
  ({ value, onChange, customInput, isDisabled }: FaceDetailerTargetPresetButtonsProps) => {
    const { t } = useTranslation();
    const [isCustomInputOpen, setIsCustomInputOpen] = useState(() => !DETAILER_TARGET_PRESET_PROMPTS.has(value));
    const isPresetValue = DETAILER_TARGET_PRESET_PROMPTS.has(value);

    useEffect(() => {
      if (!isPresetValue) {
        setIsCustomInputOpen(true);
      }
    }, [isPresetValue]);

    const onPresetClick = useCallback<MouseEventHandler<HTMLButtonElement>>(
      (event) => {
        const prompt = event.currentTarget.dataset.prompt;
        if (prompt) {
          setIsCustomInputOpen(false);
          onChange(prompt);
        }
      },
      [onChange]
    );
    const onCustomClick = useCallback(() => {
      setIsCustomInputOpen(true);
    }, []);
    const customLabel = t('parameters.faceDetailer.targetPresets.custom');

    return (
      <FormControl isDisabled={isDisabled}>
        <FormLabel>{t('parameters.faceDetailer.targetPresets.label')}</FormLabel>
        <Flex flexDir="column" gap={2} w="full">
          <Grid templateColumns={DETAILER_TARGET_BUTTON_GRID_TEMPLATE} justifyContent="space-between" gap={2} w="full">
            {DETAILER_TARGET_PRESETS.map(({ labelKey, prompt, icon: Icon }) => {
              const label = t(labelKey);
              const isSelected = !isCustomInputOpen && value === prompt;

              return (
                <Tooltip key={prompt} label={label}>
                  <IconButton
                    aria-label={label}
                    icon={<Icon size={26} />}
                    data-prompt={prompt}
                    onClick={onPresetClick}
                    isDisabled={isDisabled}
                    {...getTargetButtonStyleProps(isSelected)}
                  />
                </Tooltip>
              );
            })}
            <Tooltip label={customLabel}>
              <IconButton
                aria-label={customLabel}
                icon={<PiPencilSimpleBold size={26} />}
                onClick={onCustomClick}
                isDisabled={isDisabled}
                {...getTargetButtonStyleProps(isCustomInputOpen || !isPresetValue)}
              />
            </Tooltip>
          </Grid>
          {isCustomInputOpen && customInput}
        </Flex>
      </FormControl>
    );
  }
);

FaceDetailerTargetPresetButtons.displayName = 'FaceDetailerTargetPresetButtons';

type FaceDetailerQualityButtonsProps = {
  value: DetailerQuality;
  onChange: (value: DetailerQuality) => void;
  isDisabled?: boolean;
};

const FaceDetailerQualityButtons = memo(({ value, onChange, isDisabled }: FaceDetailerQualityButtonsProps) => {
  const { t } = useTranslation();

  const onQualityClick = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (event) => {
      const quality = event.currentTarget.dataset.quality;
      if (isInValues(quality, DETAILER_QUALITIES)) {
        onChange(quality);
      }
    },
    [onChange]
  );

  return (
    <FormControl isDisabled={isDisabled}>
      <FormLabel>{t('parameters.faceDetailer.quality')}</FormLabel>
      <ButtonGroup size="sm" variant="outline" w="full">
        {DETAILER_QUALITIES.map((quality) => (
          <Button
            key={quality}
            flex={1}
            minW={0}
            colorScheme={value === quality ? 'invokeBlue' : undefined}
            data-quality={quality}
            onClick={onQualityClick}
            isDisabled={isDisabled}
          >
            {t(`parameters.faceDetailer.qualities.${quality}`)}
          </Button>
        ))}
      </ButtonGroup>
    </FormControl>
  );
});

FaceDetailerQualityButtons.displayName = 'FaceDetailerQualityButtons';

export const FaceDetailerSettingsAccordion = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const modelSupportsFaceDetailer = useAppSelector(selectModelSupportsFaceDetailer);
  const detailerEnabled = useAppSelector(selectDetailerEnabled);
  const detailerDetector = useAppSelector(selectDetailerDetector);
  const detailerQuality = useAppSelector(selectDetailerQuality);
  const detailerFaceSelection = useAppSelector(selectDetailerFaceSelection);
  const detailerDinoModel = useAppSelector(selectDetailerDinoModel);
  const detailerSamModel = useAppSelector(selectDetailerSamModel);
  const detailerDetectionThreshold = useAppSelector(selectDetailerDetectionThreshold);
  const detailerTargetSize = useAppSelector(selectDetailerTargetSize);
  const detailerMaxUpscale = useAppSelector(selectDetailerMaxUpscale);
  const detailerMaxProcessSize = useAppSelector(selectDetailerMaxProcessSize);
  const detailerCropPadding = useAppSelector(selectDetailerCropPadding);
  const detailerDenoiseMaskExpand = useAppSelector(selectDetailerDenoiseMaskExpand);
  const detailerDenoiseMaskFeather = useAppSelector(selectDetailerDenoiseMaskFeather);
  const detailerPasteMaskExpand = useAppSelector(selectDetailerPasteMaskExpand);
  const detailerPasteMaskFeather = useAppSelector(selectDetailerPasteMaskFeather);
  const detailerColorCorrectMode = useAppSelector(selectDetailerColorCorrectMode);
  const detailerFaceId = useAppSelector(selectDetailerFaceId);
  const detailerMinConfidence = useAppSelector(selectDetailerMinConfidence);
  const detailerPadding = useAppSelector(selectDetailerPadding);
  const detailerStrength = useAppSelector(selectDetailerStrength);
  const detailerSteps = useAppSelector(selectDetailerSteps);
  const detailerCfgScale = useAppSelector(selectDetailerCfgScale);
  const detailerMaskBlur = useAppSelector(selectDetailerMaskBlur);
  const detailerTargetPrompt = useAppSelector(selectDetailerTargetPrompt);
  const isGroundedSam = detailerDetector === 'grounding-dino-sam';

  const detectorOptions = useMemo<ComboboxOption[]>(
    () => [
      { label: t('parameters.faceDetailer.detectors.groundedSam'), value: 'grounding-dino-sam' },
      { label: t('parameters.faceDetailer.detectors.mediapipe'), value: 'mediapipe' },
    ],
    [t]
  );
  const faceSelectionOptions = useMemo<ComboboxOption[]>(
    () => [
      { label: t('parameters.faceDetailer.faceSelections.highestScore'), value: 'highest_score' },
      { label: t('parameters.faceDetailer.faceSelections.largestArea'), value: 'largest_area' },
      { label: t('parameters.faceDetailer.faceSelections.index'), value: 'index' },
    ],
    [t]
  );
  const dinoModelOptions = useMemo<ComboboxOption[]>(
    () => DETAILER_DINO_MODELS.map((model) => ({ label: model, value: model })),
    []
  );
  const samModelOptions = useMemo<ComboboxOption[]>(
    () => DETAILER_SAM_MODELS.map((model) => ({ label: model, value: model })),
    []
  );
  const colorCorrectModeOptions = useMemo<ComboboxOption[]>(
    () => [
      { label: t('parameters.faceDetailer.colorCorrectModes.off'), value: 'off' },
      { label: t('parameters.faceDetailer.colorCorrectModes.luma'), value: 'YCbCr-Luma' },
      { label: t('parameters.faceDetailer.colorCorrectModes.chroma'), value: 'YCbCr-Chroma' },
      { label: t('parameters.faceDetailer.colorCorrectModes.ycbcr'), value: 'YCbCr' },
      { label: t('parameters.faceDetailer.colorCorrectModes.rgb'), value: 'RGB' },
    ],
    [t]
  );

  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: 'face-detailer-settings-generate',
    defaultIsOpen: false,
  });
  const { isOpen: isOpenAdvanced, onToggle: onToggleAdvanced } = useExpanderToggle({
    id: 'detailer-settings-advanced',
    defaultIsOpen: false,
  });
  const { isOpen: isOpenDeveloper, onToggle: onToggleDeveloper } = useExpanderToggle({
    id: 'detailer-settings-developer',
    defaultIsOpen: false,
  });
  const hasDeveloperOptions = import.meta.env.MODE === 'development';

  const onEnabledChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      dispatch(setDetailerEnabled(event.target.checked));
    },
    [dispatch]
  );
  const onDetectorChange = useCallback<ComboboxOnChange>(
    (option) => {
      if (isInValues(option?.value, DETAILER_DETECTORS)) {
        dispatch(setDetailerDetector(option.value as DetailerDetector));
      }
    },
    [dispatch]
  );
  const onQualityButtonChange = useCallback(
    (quality: DetailerQuality) => {
      dispatch(setDetailerQuality(quality));
    },
    [dispatch]
  );
  const onTargetPromptChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      dispatch(setDetailerTargetPrompt(event.target.value));
    },
    [dispatch]
  );
  const onTargetPresetChange = useCallback((prompt: string) => dispatch(setDetailerTargetPrompt(prompt)), [dispatch]);
  const onFaceSelectionChange = useCallback<ComboboxOnChange>(
    (option) => {
      if (isInValues(option?.value, DETAILER_FACE_SELECTIONS)) {
        dispatch(setDetailerFaceSelection(option.value as DetailerFaceSelection));
      }
    },
    [dispatch]
  );
  const onDinoModelChange = useCallback<ComboboxOnChange>(
    (option) => {
      if (isInValues(option?.value, DETAILER_DINO_MODELS)) {
        dispatch(setDetailerDinoModel(option.value as DetailerDinoModel));
      }
    },
    [dispatch]
  );
  const onSamModelChange = useCallback<ComboboxOnChange>(
    (option) => {
      if (isInValues(option?.value, DETAILER_SAM_MODELS)) {
        dispatch(setDetailerSamModel(option.value as DetailerSamModel));
      }
    },
    [dispatch]
  );
  const onDetectionThresholdChange = useCallback(
    (value: number) => dispatch(setDetailerDetectionThreshold(value)),
    [dispatch]
  );
  const onTargetSizeChange = useCallback((value: number) => dispatch(setDetailerTargetSize(value)), [dispatch]);
  const onMaxUpscaleChange = useCallback((value: number) => dispatch(setDetailerMaxUpscale(value)), [dispatch]);
  const onMaxProcessSizeChange = useCallback((value: number) => dispatch(setDetailerMaxProcessSize(value)), [dispatch]);
  const onCropPaddingChange = useCallback((value: number) => dispatch(setDetailerCropPadding(value)), [dispatch]);
  const onDenoiseMaskExpandChange = useCallback(
    (value: number) => dispatch(setDetailerDenoiseMaskExpand(value)),
    [dispatch]
  );
  const onDenoiseMaskFeatherChange = useCallback(
    (value: number) => dispatch(setDetailerDenoiseMaskFeather(value)),
    [dispatch]
  );
  const onPasteMaskExpandChange = useCallback(
    (value: number) => dispatch(setDetailerPasteMaskExpand(value)),
    [dispatch]
  );
  const onPasteMaskFeatherChange = useCallback(
    (value: number) => dispatch(setDetailerPasteMaskFeather(value)),
    [dispatch]
  );
  const onColorCorrectModeChange = useCallback<ComboboxOnChange>(
    (option) => {
      if (isInValues(option?.value, DETAILER_COLOR_CORRECT_MODES)) {
        dispatch(setDetailerColorCorrectMode(option.value as DetailerColorCorrectMode));
      }
    },
    [dispatch]
  );
  const onFaceIdChange = useCallback((value: number) => dispatch(setDetailerFaceId(value)), [dispatch]);
  const onMinConfidenceChange = useCallback((value: number) => dispatch(setDetailerMinConfidence(value)), [dispatch]);
  const onPaddingChange = useCallback((value: number) => dispatch(setDetailerPadding(value)), [dispatch]);
  const onStrengthChange = useCallback((value: number) => dispatch(setDetailerStrength(value)), [dispatch]);
  const onStepsChange = useCallback((value: number) => dispatch(setDetailerSteps(value)), [dispatch]);
  const onCfgScaleChange = useCallback((value: number) => dispatch(setDetailerCfgScale(value)), [dispatch]);
  const onMaskBlurChange = useCallback((value: number) => dispatch(setDetailerMaskBlur(value)), [dispatch]);

  if (!modelSupportsFaceDetailer) {
    return null;
  }

  return (
    <StandaloneAccordion
      label={t('accordions.faceDetailer.title')}
      badges={detailerEnabled ? [t('common.on')] : EMPTY_ARRAY}
      isOpen={isOpen}
      onToggle={onToggle}
    >
      <Box px={4} pt={4} pb={0} data-testid="face-detailer-settings-accordion">
        <Flex gap={4} flexDir="column" pb={0}>
          <FormControlGroup formLabelProps={compactFormLabelProps}>
            <FormControl w="min-content">
              <FormLabel>{t('parameters.faceDetailer.enabled')}</FormLabel>
              <Switch isChecked={detailerEnabled} onChange={onEnabledChange} />
            </FormControl>
            {isGroundedSam && (
              <FaceDetailerTargetPresetButtons
                value={detailerTargetPrompt}
                onChange={onTargetPresetChange}
                customInput={
                  <Input
                    aria-label={t('parameters.faceDetailer.targetPrompt')}
                    value={detailerTargetPrompt}
                    onChange={onTargetPromptChange}
                  />
                }
                isDisabled={!detailerEnabled}
              />
            )}
            <FaceDetailerQualityButtons
              value={detailerQuality}
              onChange={onQualityButtonChange}
              isDisabled={!detailerEnabled}
            />
            <FaceDetailerNumberParam
              label={t('parameters.faceDetailer.strength')}
              value={detailerStrength}
              defaultValue={STRENGTH_CONSTRAINTS.initial}
              sliderMin={STRENGTH_CONSTRAINTS.sliderMin}
              sliderMax={STRENGTH_CONSTRAINTS.sliderMax}
              numberInputMin={STRENGTH_CONSTRAINTS.numberInputMin}
              numberInputMax={STRENGTH_CONSTRAINTS.numberInputMax}
              step={STRENGTH_CONSTRAINTS.step}
              fineStep={STRENGTH_CONSTRAINTS.fineStep}
              onChange={onStrengthChange}
              isDisabled={!detailerEnabled}
            />
            <FaceDetailerNumberParam
              label={t('parameters.faceDetailer.steps')}
              value={detailerSteps}
              defaultValue={STEPS_CONSTRAINTS.initial}
              sliderMin={STEPS_CONSTRAINTS.sliderMin}
              sliderMax={STEPS_CONSTRAINTS.sliderMax}
              numberInputMin={STEPS_CONSTRAINTS.numberInputMin}
              numberInputMax={STEPS_CONSTRAINTS.numberInputMax}
              step={STEPS_CONSTRAINTS.step}
              fineStep={STEPS_CONSTRAINTS.fineStep}
              onChange={onStepsChange}
              isDisabled={!detailerEnabled}
            />
          </FormControlGroup>
        </Flex>
        <Expander label={t('accordions.advanced.options')} isOpen={isOpenAdvanced} onToggle={onToggleAdvanced}>
          <Flex gap={4} flexDir="column" pb={hasDeveloperOptions ? 0 : 4}>
            <FormControlGroup formLabelProps={formLabelProps}>
              {isGroundedSam && (
                <>
                  <FaceDetailerSelectParam
                    label={t('parameters.faceDetailer.faceSelection')}
                    value={detailerFaceSelection}
                    options={faceSelectionOptions}
                    onChange={onFaceSelectionChange}
                    isDisabled={!detailerEnabled}
                  />
                  {detailerFaceSelection === 'index' && (
                    <FaceDetailerNumberParam
                      label={t('parameters.faceDetailer.faceId')}
                      value={detailerFaceId}
                      defaultValue={FACE_ID_CONSTRAINTS.initial}
                      sliderMin={FACE_ID_CONSTRAINTS.sliderMin}
                      sliderMax={FACE_ID_CONSTRAINTS.sliderMax}
                      numberInputMin={FACE_ID_CONSTRAINTS.numberInputMin}
                      numberInputMax={FACE_ID_CONSTRAINTS.numberInputMax}
                      step={FACE_ID_CONSTRAINTS.step}
                      fineStep={FACE_ID_CONSTRAINTS.fineStep}
                      onChange={onFaceIdChange}
                      isDisabled={!detailerEnabled}
                    />
                  )}
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.detectionThreshold')}
                    value={detailerDetectionThreshold}
                    defaultValue={CONFIDENCE_CONSTRAINTS.initial}
                    sliderMin={CONFIDENCE_CONSTRAINTS.sliderMin}
                    sliderMax={CONFIDENCE_CONSTRAINTS.sliderMax}
                    numberInputMin={CONFIDENCE_CONSTRAINTS.numberInputMin}
                    numberInputMax={CONFIDENCE_CONSTRAINTS.numberInputMax}
                    step={CONFIDENCE_CONSTRAINTS.step}
                    fineStep={CONFIDENCE_CONSTRAINTS.fineStep}
                    onChange={onDetectionThresholdChange}
                    isDisabled={!detailerEnabled}
                  />
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.targetSize')}
                    value={detailerTargetSize}
                    defaultValue={TARGET_SIZE_CONSTRAINTS.initial}
                    sliderMin={TARGET_SIZE_CONSTRAINTS.sliderMin}
                    sliderMax={TARGET_SIZE_CONSTRAINTS.sliderMax}
                    numberInputMin={TARGET_SIZE_CONSTRAINTS.numberInputMin}
                    numberInputMax={TARGET_SIZE_CONSTRAINTS.numberInputMax}
                    step={TARGET_SIZE_CONSTRAINTS.step}
                    fineStep={TARGET_SIZE_CONSTRAINTS.fineStep}
                    onChange={onTargetSizeChange}
                    isDisabled={!detailerEnabled}
                  />
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.maxUpscale')}
                    value={detailerMaxUpscale}
                    defaultValue={MAX_UPSCALE_CONSTRAINTS.initial}
                    sliderMin={MAX_UPSCALE_CONSTRAINTS.sliderMin}
                    sliderMax={MAX_UPSCALE_CONSTRAINTS.sliderMax}
                    numberInputMin={MAX_UPSCALE_CONSTRAINTS.numberInputMin}
                    numberInputMax={MAX_UPSCALE_CONSTRAINTS.numberInputMax}
                    step={MAX_UPSCALE_CONSTRAINTS.step}
                    fineStep={MAX_UPSCALE_CONSTRAINTS.fineStep}
                    onChange={onMaxUpscaleChange}
                    isDisabled={!detailerEnabled}
                  />
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.maxProcessSize')}
                    value={detailerMaxProcessSize}
                    defaultValue={MAX_PROCESS_SIZE_CONSTRAINTS.initial}
                    sliderMin={MAX_PROCESS_SIZE_CONSTRAINTS.sliderMin}
                    sliderMax={MAX_PROCESS_SIZE_CONSTRAINTS.sliderMax}
                    numberInputMin={MAX_PROCESS_SIZE_CONSTRAINTS.numberInputMin}
                    numberInputMax={MAX_PROCESS_SIZE_CONSTRAINTS.numberInputMax}
                    step={MAX_PROCESS_SIZE_CONSTRAINTS.step}
                    fineStep={MAX_PROCESS_SIZE_CONSTRAINTS.fineStep}
                    onChange={onMaxProcessSizeChange}
                    isDisabled={!detailerEnabled}
                  />
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.cropPadding')}
                    value={detailerCropPadding}
                    defaultValue={CROP_PADDING_CONSTRAINTS.initial}
                    sliderMin={CROP_PADDING_CONSTRAINTS.sliderMin}
                    sliderMax={CROP_PADDING_CONSTRAINTS.sliderMax}
                    numberInputMin={CROP_PADDING_CONSTRAINTS.numberInputMin}
                    numberInputMax={CROP_PADDING_CONSTRAINTS.numberInputMax}
                    step={CROP_PADDING_CONSTRAINTS.step}
                    fineStep={CROP_PADDING_CONSTRAINTS.fineStep}
                    onChange={onCropPaddingChange}
                    isDisabled={!detailerEnabled}
                  />
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.denoiseMaskExpand')}
                    value={detailerDenoiseMaskExpand}
                    defaultValue={DENOISE_MASK_EXPAND_CONSTRAINTS.initial}
                    sliderMin={DENOISE_MASK_EXPAND_CONSTRAINTS.sliderMin}
                    sliderMax={DENOISE_MASK_EXPAND_CONSTRAINTS.sliderMax}
                    numberInputMin={DENOISE_MASK_EXPAND_CONSTRAINTS.numberInputMin}
                    numberInputMax={DENOISE_MASK_EXPAND_CONSTRAINTS.numberInputMax}
                    step={DENOISE_MASK_EXPAND_CONSTRAINTS.step}
                    fineStep={DENOISE_MASK_EXPAND_CONSTRAINTS.fineStep}
                    onChange={onDenoiseMaskExpandChange}
                    isDisabled={!detailerEnabled}
                  />
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.denoiseMaskFeather')}
                    value={detailerDenoiseMaskFeather}
                    defaultValue={DENOISE_MASK_FEATHER_CONSTRAINTS.initial}
                    sliderMin={DENOISE_MASK_FEATHER_CONSTRAINTS.sliderMin}
                    sliderMax={DENOISE_MASK_FEATHER_CONSTRAINTS.sliderMax}
                    numberInputMin={DENOISE_MASK_FEATHER_CONSTRAINTS.numberInputMin}
                    numberInputMax={DENOISE_MASK_FEATHER_CONSTRAINTS.numberInputMax}
                    step={DENOISE_MASK_FEATHER_CONSTRAINTS.step}
                    fineStep={DENOISE_MASK_FEATHER_CONSTRAINTS.fineStep}
                    onChange={onDenoiseMaskFeatherChange}
                    isDisabled={!detailerEnabled}
                  />
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.pasteMaskExpand')}
                    value={detailerPasteMaskExpand}
                    defaultValue={PASTE_MASK_EXPAND_CONSTRAINTS.initial}
                    sliderMin={PASTE_MASK_EXPAND_CONSTRAINTS.sliderMin}
                    sliderMax={PASTE_MASK_EXPAND_CONSTRAINTS.sliderMax}
                    numberInputMin={PASTE_MASK_EXPAND_CONSTRAINTS.numberInputMin}
                    numberInputMax={PASTE_MASK_EXPAND_CONSTRAINTS.numberInputMax}
                    step={PASTE_MASK_EXPAND_CONSTRAINTS.step}
                    fineStep={PASTE_MASK_EXPAND_CONSTRAINTS.fineStep}
                    onChange={onPasteMaskExpandChange}
                    isDisabled={!detailerEnabled}
                  />
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.pasteMaskFeather')}
                    value={detailerPasteMaskFeather}
                    defaultValue={PASTE_MASK_FEATHER_CONSTRAINTS.initial}
                    sliderMin={PASTE_MASK_FEATHER_CONSTRAINTS.sliderMin}
                    sliderMax={PASTE_MASK_FEATHER_CONSTRAINTS.sliderMax}
                    numberInputMin={PASTE_MASK_FEATHER_CONSTRAINTS.numberInputMin}
                    numberInputMax={PASTE_MASK_FEATHER_CONSTRAINTS.numberInputMax}
                    step={PASTE_MASK_FEATHER_CONSTRAINTS.step}
                    fineStep={PASTE_MASK_FEATHER_CONSTRAINTS.fineStep}
                    onChange={onPasteMaskFeatherChange}
                    isDisabled={!detailerEnabled}
                  />
                  <FaceDetailerSelectParam
                    label={t('parameters.faceDetailer.colorCorrectMode')}
                    value={detailerColorCorrectMode}
                    options={colorCorrectModeOptions}
                    onChange={onColorCorrectModeChange}
                    isDisabled={!detailerEnabled}
                  />
                </>
              )}
              <FaceDetailerNumberParam
                label={t('parameters.faceDetailer.cfgScale')}
                value={detailerCfgScale}
                defaultValue={CFG_SCALE_CONSTRAINTS.initial}
                sliderMin={CFG_SCALE_CONSTRAINTS.sliderMin}
                sliderMax={CFG_SCALE_CONSTRAINTS.sliderMax}
                numberInputMin={CFG_SCALE_CONSTRAINTS.numberInputMin}
                numberInputMax={CFG_SCALE_CONSTRAINTS.numberInputMax}
                step={CFG_SCALE_CONSTRAINTS.step}
                fineStep={CFG_SCALE_CONSTRAINTS.fineStep}
                onChange={onCfgScaleChange}
                isDisabled={!detailerEnabled}
              />
            </FormControlGroup>
            {hasDeveloperOptions && (
              <Expander
                label={t('parameters.faceDetailer.developerOptions')}
                isOpen={isOpenDeveloper}
                onToggle={onToggleDeveloper}
              >
                <Flex gap={4} flexDir="column" pb={4}>
                  <FormControlGroup formLabelProps={formLabelProps}>
                    <FaceDetailerSelectParam
                      label={t('parameters.faceDetailer.detector')}
                      value={detailerDetector}
                      options={detectorOptions}
                      onChange={onDetectorChange}
                      isDisabled={!detailerEnabled}
                    />
                    {isGroundedSam && (
                      <>
                        <FaceDetailerSelectParam
                          label={t('parameters.faceDetailer.detectorModel')}
                          value={detailerDinoModel}
                          options={dinoModelOptions}
                          onChange={onDinoModelChange}
                          isDisabled={!detailerEnabled}
                        />
                        <FaceDetailerSelectParam
                          label={t('parameters.faceDetailer.samModel')}
                          value={detailerSamModel}
                          options={samModelOptions}
                          onChange={onSamModelChange}
                          isDisabled={!detailerEnabled}
                        />
                      </>
                    )}
                    {!isGroundedSam && (
                      <>
                        <FaceDetailerNumberParam
                          label={t('parameters.faceDetailer.faceId')}
                          value={detailerFaceId}
                          defaultValue={FACE_ID_CONSTRAINTS.initial}
                          sliderMin={FACE_ID_CONSTRAINTS.sliderMin}
                          sliderMax={FACE_ID_CONSTRAINTS.sliderMax}
                          numberInputMin={FACE_ID_CONSTRAINTS.numberInputMin}
                          numberInputMax={FACE_ID_CONSTRAINTS.numberInputMax}
                          step={FACE_ID_CONSTRAINTS.step}
                          fineStep={FACE_ID_CONSTRAINTS.fineStep}
                          onChange={onFaceIdChange}
                          isDisabled={!detailerEnabled}
                        />
                        <FaceDetailerNumberParam
                          label={t('parameters.faceDetailer.minConfidence')}
                          value={detailerMinConfidence}
                          defaultValue={CONFIDENCE_CONSTRAINTS.initial}
                          sliderMin={CONFIDENCE_CONSTRAINTS.sliderMin}
                          sliderMax={CONFIDENCE_CONSTRAINTS.sliderMax}
                          numberInputMin={CONFIDENCE_CONSTRAINTS.numberInputMin}
                          numberInputMax={CONFIDENCE_CONSTRAINTS.numberInputMax}
                          step={CONFIDENCE_CONSTRAINTS.step}
                          fineStep={CONFIDENCE_CONSTRAINTS.fineStep}
                          onChange={onMinConfidenceChange}
                          isDisabled={!detailerEnabled}
                        />
                        <FaceDetailerNumberParam
                          label={t('parameters.faceDetailer.padding')}
                          value={detailerPadding}
                          defaultValue={PADDING_CONSTRAINTS.initial}
                          sliderMin={PADDING_CONSTRAINTS.sliderMin}
                          sliderMax={PADDING_CONSTRAINTS.sliderMax}
                          numberInputMin={PADDING_CONSTRAINTS.numberInputMin}
                          numberInputMax={PADDING_CONSTRAINTS.numberInputMax}
                          step={PADDING_CONSTRAINTS.step}
                          fineStep={PADDING_CONSTRAINTS.fineStep}
                          onChange={onPaddingChange}
                          isDisabled={!detailerEnabled}
                        />
                        <FaceDetailerNumberParam
                          label={t('parameters.faceDetailer.maskBlur')}
                          value={detailerMaskBlur}
                          defaultValue={MASK_BLUR_CONSTRAINTS.initial}
                          sliderMin={MASK_BLUR_CONSTRAINTS.sliderMin}
                          sliderMax={MASK_BLUR_CONSTRAINTS.sliderMax}
                          numberInputMin={MASK_BLUR_CONSTRAINTS.numberInputMin}
                          numberInputMax={MASK_BLUR_CONSTRAINTS.numberInputMax}
                          step={MASK_BLUR_CONSTRAINTS.step}
                          fineStep={MASK_BLUR_CONSTRAINTS.fineStep}
                          onChange={onMaskBlurChange}
                          isDisabled={!detailerEnabled}
                        />
                      </>
                    )}
                  </FormControlGroup>
                </Flex>
              </Expander>
            )}
          </Flex>
        </Expander>
      </Box>
    </StandaloneAccordion>
  );
});

FaceDetailerSettingsAccordion.displayName = 'FaceDetailerSettingsAccordion';
