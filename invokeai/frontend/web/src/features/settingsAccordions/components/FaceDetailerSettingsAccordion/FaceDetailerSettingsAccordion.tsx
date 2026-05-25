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
import type { Feature } from 'common/components/InformationalPopover/constants';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { isDetailerQualityPresetAdjusted } from 'features/controlLayers/store/detailerQualityPresets';
import type { GroundedSamDetailerRuntimeConfig } from 'features/controlLayers/store/detailerRuntimeConfig';
import { getGroundedSamDetailerRuntimeConfig } from 'features/controlLayers/store/detailerRuntimeConfig';
import {
  selectDetailerCfgScale,
  selectDetailerColorCorrectMode,
  selectDetailerCropPadding,
  selectDetailerDebugEnabled,
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
  setDetailerDebugEnabled,
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
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiHandBold, PiHeadCircuitBold, PiPencilSimpleBold, PiPersonBold, PiSmileyBold } from 'react-icons/pi';

import {
  DETAILER_DINO_MODELS,
  DETAILER_SAM_MODELS,
  getDetailerDinoModelOptions,
  getDetailerSamModelOptions,
} from './detailerModelOptions';

const DETAILER_LABEL_WIDTH = '6rem';

const formLabelProps: FormLabelProps = {
  lineHeight: 'shorter',
  maxW: DETAILER_LABEL_WIDTH,
  minW: DETAILER_LABEL_WIDTH,
  whiteSpace: 'normal',
};

const DETAILER_DETECTORS = ['grounding-dino-sam', 'mediapipe'] as const;
const DETAILER_QUALITIES = ['fast', 'balanced', 'high'] as const;
const DETAILER_FACE_SELECTIONS = ['highest_score', 'largest_area', 'index'] as const;
const DETAILER_COLOR_CORRECT_MODES = ['off', 'YCbCr-Luma', 'YCbCr-Chroma', 'YCbCr', 'RGB'] as const;

const DETAILER_TARGET_PRESETS = [
  {
    labelKey: 'parameters.faceDetailer.targetPresets.face',
    tooltipKey: 'parameters.faceDetailer.targetPresets.face',
    prompt: 'face',
    icon: PiSmileyBold,
  },
  {
    labelKey: 'parameters.faceDetailer.targetPresets.head',
    tooltipKey: 'parameters.faceDetailer.targetPresets.head',
    prompt: 'head',
    icon: PiHeadCircuitBold,
  },
  {
    labelKey: 'parameters.faceDetailer.targetPresets.hands',
    tooltipKey: 'parameters.faceDetailer.targetPresets.hands',
    prompt: 'hands',
    icon: PiHandBold,
  },
  {
    labelKey: 'parameters.faceDetailer.targetPresets.body',
    tooltipKey: 'parameters.faceDetailer.targetPresets.body',
    prompt: 'person',
    icon: PiPersonBold,
  },
] as const;

const DETAILER_TARGET_PRESET_PROMPTS = new Set<string>(DETAILER_TARGET_PRESETS.map(({ prompt }) => prompt));
const DETAILER_TARGET_BUTTON_GRID_TEMPLATE = 'repeat(5, minmax(2.5rem, 3.5rem))';
const DETAILER_SUPPORTED_DETECTOR: DetailerDetector = 'grounding-dino-sam';

const customTargetInputStyleProps = {
  bg: 'base.800',
  borderColor: 'base.600',
  color: 'base.50',
  _focusVisible: {
    bg: 'base.800',
    borderColor: 'invokeBlue.400',
    boxShadow: '0 0 0 1px var(--invoke-colors-invokeBlue-400)',
  },
  _hover: {
    bg: 'base.800',
    borderColor: 'base.500',
  },
  sx: {
    '&:-webkit-autofill, &:-webkit-autofill:hover, &:-webkit-autofill:focus': {
      WebkitBoxShadow: '0 0 0 1000px var(--invoke-colors-base-800) inset',
      WebkitTextFillColor: 'var(--invoke-colors-base-50)',
      caretColor: 'var(--invoke-colors-base-50)',
      transition: 'background-color 9999s ease-in-out 0s',
    },
  },
} as const;

const getTargetButtonStyleProps = (isSelected: boolean) => ({
  aspectRatio: 1,
  bg: 'base.800',
  borderColor: isSelected ? 'base.500' : 'base.600',
  borderRadius: 'base',
  borderStyle: 'solid',
  borderWidth: 1,
  boxShadow: 'none',
  color: isSelected ? 'base.50' : 'base.300',
  h: 'auto',
  minH: 13,
  minW: 0,
  overflow: 'hidden',
  position: 'relative' as const,
  variant: 'ghost',
  w: 'full',
  _after: {
    bg: isSelected ? 'invokeBlue.400' : 'transparent',
    borderRadius: 'full',
    bottom: 0,
    content: '""',
    h: 0.5,
    insetInlineEnd: 2,
    insetInlineStart: 2,
    pointerEvents: 'none',
    position: 'absolute' as const,
  },
  _active: {
    bg: 'base.700',
  },
  _disabled: {
    cursor: 'not-allowed',
    opacity: 0.4,
  },
  _hover: {
    bg: 'base.700',
    borderColor: isSelected ? 'base.400' : 'base.500',
    color: 'base.50',
    _after: {
      bg: isSelected ? 'invokeBlue.300' : 'transparent',
    },
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
  initial: 48,
  sliderMin: 0,
  sliderMax: 256,
  numberInputMin: 0,
  numberInputMax: 512,
  step: 8,
  fineStep: 1,
};

const STRENGTH_CONSTRAINTS = {
  initial: 0.26,
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
  initial: 4.5,
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
  initial: 1024,
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
  initial: 6,
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
  initial: 6,
  sliderMin: 0,
  sliderMax: 96,
  numberInputMin: 0,
  numberInputMax: 256,
  step: 1,
  fineStep: 1,
};

const isInValues = <T extends string>(value: unknown, values: readonly T[]): value is T =>
  typeof value === 'string' && values.includes(value as T);

const formatEffectiveValue = (value: number) =>
  Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.?0+$/, '');

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
  effectiveValueLabel?: string;
  popoverFeature?: Feature;
};

const FaceDetailerLabelText = memo(({ label, popoverFeature }: { label: string; popoverFeature?: Feature }) => {
  const labelText = (
    <Box as="span" cursor={popoverFeature ? 'help' : undefined}>
      {label}
    </Box>
  );

  if (!popoverFeature) {
    return labelText;
  }

  return <InformationalPopover feature={popoverFeature}>{labelText}</InformationalPopover>;
});

FaceDetailerLabelText.displayName = 'FaceDetailerLabelText';

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
    effectiveValueLabel,
    popoverFeature,
  }: FaceDetailerNumberParamProps) => {
    return (
      <FormControl isDisabled={isDisabled}>
        <FormLabel>
          <Flex flexDir="column" gap={1}>
            <FaceDetailerLabelText label={label} popoverFeature={popoverFeature} />
            {effectiveValueLabel && (
              <Box as="span" color="invokeBlue.300" fontSize="xs" fontWeight="semibold" lineHeight="short">
                {effectiveValueLabel}
              </Box>
            )}
          </Flex>
        </FormLabel>
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
    );
  }
);

FaceDetailerNumberParam.displayName = 'FaceDetailerNumberParam';

type FaceDetailerSelectParamProps = {
  label: string;
  value: string;
  options: ComboboxOption[];
  onChange: ComboboxOnChange;
  isDisabled?: boolean;
  popoverFeature?: Feature;
};

const FaceDetailerSelectParam = memo(
  ({ label, value, options, onChange, isDisabled, popoverFeature }: FaceDetailerSelectParamProps) => {
    const selected = useMemo(() => options.find((option) => option.value === value), [options, value]);

    return (
      <FormControl isDisabled={isDisabled}>
        <FormLabel>
          <FaceDetailerLabelText label={label} popoverFeature={popoverFeature} />
        </FormLabel>
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
    const previousValueRef = useRef(value);
    const isPresetValue = DETAILER_TARGET_PRESET_PROMPTS.has(value);

    useEffect(() => {
      if (previousValueRef.current === value) {
        return;
      }

      previousValueRef.current = value;
      setIsCustomInputOpen(!DETAILER_TARGET_PRESET_PROMPTS.has(value));
    }, [value]);

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
            {DETAILER_TARGET_PRESETS.map(({ labelKey, tooltipKey, prompt, icon: Icon }) => {
              const label = t(labelKey);
              const tooltip = t(tooltipKey ?? labelKey);
              const isSelected = !isCustomInputOpen && value === prompt;

              return (
                <Tooltip key={prompt} label={tooltip}>
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

type FaceDetailerAdvancedSectionProps = {
  label: string;
  children: ReactNode;
};

const FaceDetailerAdvancedSection = memo(({ label, children }: FaceDetailerAdvancedSectionProps) => {
  return (
    <Flex flexDir="column" gap={3}>
      <Flex alignItems="center" gap={3} color="base.500">
        <Box as="span" flexShrink={0} fontSize="sm" fontWeight="semibold">
          {label}
        </Box>
        <Box borderTopColor="base.700" borderTopStyle="solid" borderTopWidth={1} flex={1} />
      </Flex>
      <FormControlGroup formLabelProps={formLabelProps}>{children}</FormControlGroup>
    </Flex>
  );
});

FaceDetailerAdvancedSection.displayName = 'FaceDetailerAdvancedSection';

type FaceDetailerQualityButtonsProps = {
  value: DetailerQuality;
  onChange: (value: DetailerQuality) => void;
  isAdjusted: boolean;
  isDisabled?: boolean;
};

const FaceDetailerQualityButtons = memo(
  ({ value, onChange, isAdjusted, isDisabled }: FaceDetailerQualityButtonsProps) => {
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
        <FormLabel>
          <Flex alignItems="center" gap={2}>
            <FaceDetailerLabelText label={t('parameters.faceDetailer.quality')} popoverFeature="detailerQuality" />
            {isAdjusted && (
              <Box as="span" color="invokeBlue.300" fontSize="xs" fontWeight="semibold">
                {t('parameters.faceDetailer.adjusted')}
              </Box>
            )}
          </Flex>
        </FormLabel>
        <ButtonGroup size="sm" variant="outline" w="full">
          {DETAILER_QUALITIES.map((quality) => {
            const label = t(`parameters.faceDetailer.qualities.${quality}`);

            return (
              <Button
                key={quality}
                flex={1}
                minW={0}
                colorScheme={value === quality ? 'invokeBlue' : undefined}
                data-quality={quality}
                onClick={onQualityClick}
                isDisabled={isDisabled}
              >
                {label}
              </Button>
            );
          })}
        </ButtonGroup>
      </FormControl>
    );
  }
);

FaceDetailerQualityButtons.displayName = 'FaceDetailerQualityButtons';

type FaceDetailerEffectiveProfileSummaryProps = {
  runtimeConfig: GroundedSamDetailerRuntimeConfig;
  isDisabled?: boolean;
};

const FaceDetailerEffectiveProfileSummary = memo(
  ({ runtimeConfig, isDisabled }: FaceDetailerEffectiveProfileSummaryProps) => {
    const { t } = useTranslation();

    if (runtimeConfig.targetProfile !== 'person') {
      return null;
    }

    return (
      <FormControl isDisabled={isDisabled}>
        <FormLabel>{t('parameters.faceDetailer.effectiveProfile')}</FormLabel>
        <Flex
          alignItems="center"
          gap={2}
          bg="base.800"
          borderColor="base.700"
          borderRadius="base"
          borderStyle="solid"
          borderWidth={1}
          color="base.300"
          fontSize="sm"
          lineHeight="short"
          minH={10}
          px={3}
          py={2}
        >
          <Box as="span" color="invokeBlue.300" fontWeight="semibold">
            {t('parameters.faceDetailer.bodyProfileActive')}
          </Box>
        </Flex>
      </FormControl>
    );
  }
);

FaceDetailerEffectiveProfileSummary.displayName = 'FaceDetailerEffectiveProfileSummary';

export const FaceDetailerSettingsAccordion = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const modelSupportsFaceDetailer = useAppSelector(selectModelSupportsFaceDetailer);
  const detailerEnabled = useAppSelector(selectDetailerEnabled);
  const detailerDetector = useAppSelector(selectDetailerDetector);
  const detailerDebugEnabled = useAppSelector(selectDetailerDebugEnabled);
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
  const isDetailerQualityAdjusted = useMemo(
    () =>
      isDetailerQualityPresetAdjusted({
        detailerQuality,
        detailerTargetSize,
        detailerMaxUpscale,
        detailerMaxProcessSize,
        detailerCropPadding,
        detailerStrength,
        detailerSteps,
        detailerCfgScale,
        detailerDenoiseMaskFeather,
        detailerPasteMaskExpand,
        detailerPasteMaskFeather,
        detailerSamModel,
      }),
    [
      detailerCfgScale,
      detailerCropPadding,
      detailerDenoiseMaskFeather,
      detailerMaxProcessSize,
      detailerMaxUpscale,
      detailerPasteMaskExpand,
      detailerPasteMaskFeather,
      detailerQuality,
      detailerSamModel,
      detailerSteps,
      detailerStrength,
      detailerTargetSize,
    ]
  );
  const detailerRuntimeConfig = useMemo(
    () =>
      getGroundedSamDetailerRuntimeConfig(
        {
          detailerQuality,
          detailerTargetSize,
          detailerMaxUpscale,
          detailerMaxProcessSize,
          detailerDenoiseMaskExpand,
          detailerDenoiseMaskFeather,
          detailerPasteMaskExpand,
          detailerPasteMaskFeather,
          detailerCfgScale,
          detailerSteps,
          detailerStrength,
        },
        detailerTargetPrompt
      ),
    [
      detailerCfgScale,
      detailerDenoiseMaskExpand,
      detailerDenoiseMaskFeather,
      detailerMaxProcessSize,
      detailerMaxUpscale,
      detailerPasteMaskExpand,
      detailerPasteMaskFeather,
      detailerQuality,
      detailerSteps,
      detailerStrength,
      detailerTargetPrompt,
      detailerTargetSize,
    ]
  );
  const shouldShowEffectiveValues = isGroundedSam && detailerRuntimeConfig.targetProfile === 'person';
  const getEffectiveValueLabel = useCallback(
    (storedValue: number, effectiveValue: number) => {
      if (!shouldShowEffectiveValues || storedValue === effectiveValue) {
        return;
      }

      return t('parameters.faceDetailer.effectiveValue', { value: formatEffectiveValue(effectiveValue) });
    },
    [shouldShowEffectiveValues, t]
  );

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
  const dinoModelOptions = useMemo<ComboboxOption[]>(() => getDetailerDinoModelOptions(t), [t]);
  const samModelOptions = useMemo<ComboboxOption[]>(() => getDetailerSamModelOptions(t), [t]);
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
  useEffect(() => {
    if (!hasDeveloperOptions && detailerDetector !== DETAILER_SUPPORTED_DETECTOR) {
      dispatch(setDetailerDetector(DETAILER_SUPPORTED_DETECTOR));
    }
  }, [detailerDetector, dispatch, hasDeveloperOptions]);
  const detailerTargetBadge = useMemo(() => {
    const preset = DETAILER_TARGET_PRESETS.find(({ prompt }) => prompt === detailerTargetPrompt);
    return preset ? t(preset.labelKey) : t('parameters.faceDetailer.targetPresets.custom');
  }, [detailerTargetPrompt, t]);
  const detailerBadges = useMemo(
    () =>
      detailerEnabled
        ? [t('common.on'), detailerTargetBadge, t(`parameters.faceDetailer.qualities.${detailerQuality}`)]
        : EMPTY_ARRAY,
    [detailerEnabled, detailerQuality, detailerTargetBadge, t]
  );

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
  const onDebugEnabledChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      dispatch(setDetailerDebugEnabled(event.target.checked));
    },
    [dispatch]
  );
  const onQualityButtonChange = useCallback(
    (quality: DetailerQuality) => {
      if (quality !== detailerQuality) {
        dispatch(setDetailerQuality(quality));
      }
    },
    [detailerQuality, dispatch]
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
      badges={detailerBadges}
      isOpen={isOpen}
      onToggle={onToggle}
    >
      <Box px={4} pt={4} pb={0} data-testid="face-detailer-settings-accordion">
        <Flex gap={4} flexDir="column" pb={0}>
          <FormControlGroup formLabelProps={formLabelProps}>
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
                    {...customTargetInputStyleProps}
                  />
                }
                isDisabled={!detailerEnabled}
              />
            )}
            <FaceDetailerQualityButtons
              value={detailerQuality}
              onChange={onQualityButtonChange}
              isAdjusted={isDetailerQualityAdjusted}
              isDisabled={!detailerEnabled}
            />
            {isGroundedSam && (
              <FaceDetailerEffectiveProfileSummary
                runtimeConfig={detailerRuntimeConfig}
                isDisabled={!detailerEnabled}
              />
            )}
            <FaceDetailerNumberParam
              label={t('parameters.faceDetailer.strength')}
              popoverFeature="detailerDenoisingStrength"
              value={detailerStrength}
              effectiveValueLabel={getEffectiveValueLabel(detailerStrength, detailerRuntimeConfig.strength)}
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
              popoverFeature="detailerSteps"
              value={detailerSteps}
              effectiveValueLabel={getEffectiveValueLabel(detailerSteps, detailerRuntimeConfig.steps)}
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
            {isGroundedSam && (
              <>
                <FaceDetailerAdvancedSection label={t('parameters.faceDetailer.advancedGroups.detection')}>
                  <FaceDetailerSelectParam
                    label={t('parameters.faceDetailer.faceSelection')}
                    popoverFeature="detailerTargetSelection"
                    value={detailerFaceSelection}
                    options={faceSelectionOptions}
                    onChange={onFaceSelectionChange}
                    isDisabled={!detailerEnabled}
                  />
                  {detailerFaceSelection === 'index' && (
                    <FaceDetailerNumberParam
                      label={t('parameters.faceDetailer.faceId')}
                      popoverFeature="detailerTargetId"
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
                    popoverFeature="detailerDetectionThreshold"
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
                  <FaceDetailerSelectParam
                    label={t('parameters.faceDetailer.detectorModel')}
                    popoverFeature="detailerDinoModel"
                    value={detailerDinoModel}
                    options={dinoModelOptions}
                    onChange={onDinoModelChange}
                    isDisabled={!detailerEnabled}
                  />
                  <FaceDetailerSelectParam
                    label={t('parameters.faceDetailer.samModel')}
                    popoverFeature="detailerSamModel"
                    value={detailerSamModel}
                    options={samModelOptions}
                    onChange={onSamModelChange}
                    isDisabled={!detailerEnabled}
                  />
                </FaceDetailerAdvancedSection>
                <FaceDetailerAdvancedSection label={t('parameters.faceDetailer.advancedGroups.cropScale')}>
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.targetSize')}
                    popoverFeature="detailerTargetSize"
                    value={detailerTargetSize}
                    effectiveValueLabel={getEffectiveValueLabel(detailerTargetSize, detailerRuntimeConfig.targetSize)}
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
                    popoverFeature="detailerMaxUpscale"
                    value={detailerMaxUpscale}
                    effectiveValueLabel={getEffectiveValueLabel(detailerMaxUpscale, detailerRuntimeConfig.maxUpscale)}
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
                    popoverFeature="detailerMaxProcess"
                    value={detailerMaxProcessSize}
                    effectiveValueLabel={getEffectiveValueLabel(
                      detailerMaxProcessSize,
                      detailerRuntimeConfig.maxProcessSize
                    )}
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
                    popoverFeature="detailerCropPadding"
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
                </FaceDetailerAdvancedSection>
                <FaceDetailerAdvancedSection label={t('parameters.faceDetailer.advancedGroups.masksPaste')}>
                  <FaceDetailerNumberParam
                    label={t('parameters.faceDetailer.denoiseMaskExpand')}
                    popoverFeature="detailerDenoiseMaskExpand"
                    value={detailerDenoiseMaskExpand}
                    effectiveValueLabel={getEffectiveValueLabel(
                      detailerDenoiseMaskExpand,
                      detailerRuntimeConfig.denoiseMaskExpand
                    )}
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
                    popoverFeature="detailerDenoiseMaskFeather"
                    value={detailerDenoiseMaskFeather}
                    effectiveValueLabel={getEffectiveValueLabel(
                      detailerDenoiseMaskFeather,
                      detailerRuntimeConfig.denoiseMaskFeather
                    )}
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
                    popoverFeature="detailerPasteMaskExpand"
                    value={detailerPasteMaskExpand}
                    effectiveValueLabel={getEffectiveValueLabel(
                      detailerPasteMaskExpand,
                      detailerRuntimeConfig.pasteMaskExpand
                    )}
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
                    popoverFeature="detailerPasteMaskFeather"
                    value={detailerPasteMaskFeather}
                    effectiveValueLabel={getEffectiveValueLabel(
                      detailerPasteMaskFeather,
                      detailerRuntimeConfig.pasteMaskFeather
                    )}
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
                    popoverFeature="detailerColorMatch"
                    value={detailerColorCorrectMode}
                    options={colorCorrectModeOptions}
                    onChange={onColorCorrectModeChange}
                    isDisabled={!detailerEnabled}
                  />
                </FaceDetailerAdvancedSection>
              </>
            )}
            <FaceDetailerAdvancedSection label={t('parameters.faceDetailer.advancedGroups.denoise')}>
              <FaceDetailerNumberParam
                label={t('parameters.faceDetailer.cfgScale')}
                popoverFeature="detailerCfgScale"
                value={detailerCfgScale}
                effectiveValueLabel={getEffectiveValueLabel(detailerCfgScale, detailerRuntimeConfig.cfgScale)}
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
            </FaceDetailerAdvancedSection>
            {hasDeveloperOptions && (
              <Expander
                label={t('parameters.faceDetailer.developerOptions')}
                isOpen={isOpenDeveloper}
                onToggle={onToggleDeveloper}
              >
                <Flex gap={4} flexDir="column" pb={4}>
                  <FormControlGroup formLabelProps={formLabelProps}>
                    <FormControl isDisabled={!detailerEnabled || !isGroundedSam} w="min-content">
                      <FormLabel>{t('parameters.faceDetailer.debugOutput')}</FormLabel>
                      <Switch isChecked={detailerDebugEnabled} onChange={onDebugEnabledChange} />
                    </FormControl>
                    <FaceDetailerSelectParam
                      label={t('parameters.faceDetailer.detector')}
                      popoverFeature="detailerDetector"
                      value={detailerDetector}
                      options={detectorOptions}
                      onChange={onDetectorChange}
                      isDisabled={!detailerEnabled}
                    />
                    {!isGroundedSam && (
                      <>
                        <FormControl isDisabled={!detailerEnabled}>
                          <FormLabel>{t('parameters.faceDetailer.legacyDetector')}</FormLabel>
                          <Box
                            bg="base.800"
                            borderColor="base.700"
                            borderRadius="base"
                            borderStyle="solid"
                            borderWidth={1}
                            color="base.300"
                            fontSize="sm"
                            lineHeight="short"
                            px={3}
                            py={2}
                          >
                            {t('parameters.faceDetailer.mediapipeLegacyWarning')}
                          </Box>
                        </FormControl>
                        <FaceDetailerNumberParam
                          label={t('parameters.faceDetailer.faceId')}
                          popoverFeature="detailerTargetId"
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
                          popoverFeature="detailerMediapipeConfidence"
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
                          popoverFeature="detailerMediapipePadding"
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
                          popoverFeature="detailerMediapipeMaskBlur"
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
