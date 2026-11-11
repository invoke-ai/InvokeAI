import type { SelectValueChangeDetails, SliderValueChangeDetails } from '@chakra-ui/react';
import type {
  ClipVisionModel,
  FluxReduxImageInfluence,
  GenerateReferenceImageConfig,
  IPAdapterMethod,
} from '@features/generation/core/types';
import type { FeatureHintId } from '@platform/ui/hints';
import type { ReactNode } from 'react';

import { Box, chakra, createListCollection, HStack, SegmentGroup, Stack, Text } from '@chakra-ui/react';
import { DEFAULT_KREA2_STYLE_STRENGTH, MAX_KREA2_STYLE_STRENGTH } from '@features/generation/core/settings';
import { Field, FieldLabel, INSIDE_LABEL_INSET } from '@platform/ui/Field';
import { FeatureHint } from '@platform/ui/hints';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Select } from '@platform/ui/Select';
import { Slider } from '@platform/ui/Slider';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { StyleMethod } from './referenceImageConfig';

import {
  CLIP_VISION_MODELS,
  formatPct,
  formatWeight,
  getReferenceMode,
  isClipVisionModel,
  isReferenceMode,
  isStyleMethod,
  MODE_SEGMENTS,
  STYLE_VARIANT_OPTIONS,
} from './referenceImageConfig';

type IPAdapterConfig = Extract<GenerateReferenceImageConfig, { type: 'ip_adapter' }>;
type Krea2ReferenceConfig = Extract<
  GenerateReferenceImageConfig,
  { type: 'krea2_reference_image' | 'krea2_edit_reference_image' }
>;
type ClipVisionOption = { label: ClipVisionModel; value: ClipVisionModel };
type StyleVariantOption = { description: string; label: string; value: StyleMethod };
type FluxReduxImageInfluenceOption = { label: string; value: FluxReduxImageInfluence };

const CLIP_VISION_COLLECTION = createListCollection<ClipVisionOption>({
  items: CLIP_VISION_MODELS.map((model) => ({ label: model, value: model })),
});

const FLUX_REDUX_IMAGE_INFLUENCE_OPTIONS: FluxReduxImageInfluenceOption[] = [
  { label: 'Lowest', value: 'lowest' },
  { label: 'Low', value: 'low' },
  { label: 'Medium', value: 'medium' },
  { label: 'High', value: 'high' },
  { label: 'Highest', value: 'highest' },
];

const FLUX_REDUX_IMAGE_INFLUENCE_COLLECTION = createListCollection<FluxReduxImageInfluenceOption>({
  items: FLUX_REDUX_IMAGE_INFLUENCE_OPTIONS,
});

const isFluxReduxImageInfluence = (value: unknown): value is FluxReduxImageInfluence =>
  FLUX_REDUX_IMAGE_INFLUENCE_OPTIONS.some((option) => option.value === value);

const renderStyleVariantItem = (option: StyleVariantOption) => (
  <Stack as="span" gap="0.5" py="0.5">
    <Text as="span" fontSize="md">
      {option.label}
    </Text>
    <Text as="span" color="fg.muted" fontSize="xs" lineHeight="short">
      {option.description}
    </Text>
  </Stack>
);

/** The default weight is the one stop a drag can snap to; the track ends stand in for 0 and 2. */
const WEIGHT_MARKS = [0, 1, 2];
export const DEFAULT_REFERENCE_WEIGHT = 1;

const STEPS_FIELD_DISABLED_CSS = { opacity: 0.5 } as const;

/**
 * Controls on a list row are backed with the settings section's surface, as a concept row backs its weight: the row's
 * hover tint would otherwise wash out their transparent fills and faint borders, and their own hover fills with it.
 */
export const ControlSurface = ({ children, ...props }: { children: ReactNode } & Record<`data-${string}`, string>) => (
  <Box bg="bg.muted" data-control-surface="" flex="1" minW="0" rounded="control" {...props}>
    {children}
  </Box>
);

/** Tiny muted label row above a control, with optional right-aligned content. */
const FieldHeader = ({ children, hint, label }: { children?: ReactNode; hint?: FeatureHintId; label: string }) => {
  const labelText = (
    <Text color="fg.muted" fontSize="xs" fontWeight="medium">
      {label}
    </Text>
  );

  return (
    <HStack justify="space-between" minH="4">
      {hint ? <FeatureHint hint={hint}>{labelText}</FeatureHint> : labelText}
      {children}
    </HStack>
  );
};

/**
 * The adapter's weight, its one control on the row. `strengthOnly` adapters (Anima) take a strength alone, which
 * cannot go negative.
 */
export const IPAdapterWeightField = ({
  config,
  disabled,
  onChange,
  strengthOnly = false,
}: {
  config: IPAdapterConfig;
  disabled: boolean;
  onChange: (config: GenerateReferenceImageConfig) => void;
  strengthOnly?: boolean;
}) => {
  const { t } = useTranslation();
  // Do not compare against captured drag-start values; returning to the start still needs a commit.
  const handleWeightChange = useCallback((weight: number) => onChange({ ...config, weight }), [config, onChange]);

  return (
    <ScrubberField
      defaultValue={DEFAULT_REFERENCE_WEIGHT}
      disabled={disabled}
      formatValue={formatWeight}
      hint={strengthOnly ? 'referenceImageStrength' : 'referenceImageWeight'}
      inputMin={strengthOnly ? 0 : -1}
      label={strengthOnly ? t('widgets.generate.strength') : t('widgets.generate.weight')}
      marks={WEIGHT_MARKS}
      max={2}
      min={0}
      step={0.05}
      value={config.weight}
      onChange={handleWeightChange}
    />
  );
};

/**
 * The adapter's settings under its row: mode (with the style variant for Style), the active step window, then
 * `children` (the model pickers). `strengthOnly` adapters (Anima) have no mode.
 */
export const IPAdapterSettings = ({
  children,
  config,
  disabled,
  onChange,
  strengthOnly = false,
}: {
  children?: ReactNode;
  config: IPAdapterConfig;
  disabled: boolean;
  onChange: (config: GenerateReferenceImageConfig) => void;
  strengthOnly?: boolean;
}) => {
  const { t } = useTranslation();
  const mode = getReferenceMode(config.method);
  const styleVariantCollection = useMemo(
    () =>
      createListCollection<StyleVariantOption>({
        items: STYLE_VARIANT_OPTIONS.map((option) => ({
          description: t(option.descriptionKey),
          label: t(option.labelKey),
          value: option.value,
        })),
      }),
    [t]
  );
  const styleVariantValue = useMemo<StyleMethod[]>(
    () => [isStyleMethod(config.method) ? config.method : 'style'],
    [config.method]
  );

  const handleModeChange = useCallback(
    ({ value }: SegmentGroup.ValueChangeDetails) => {
      if (!isReferenceMode(value)) {
        return;
      }

      const nextMethod: IPAdapterMethod =
        value === 'style' ? (isStyleMethod(config.method) ? config.method : 'style') : value;

      if (nextMethod !== config.method) {
        onChange({ ...config, method: nextMethod });
      }
    },
    [config, onChange]
  );

  const handleStyleVariantChange = useCallback(
    ({ value }: SelectValueChangeDetails<StyleVariantOption>) => {
      const nextMethod = value[0] as IPAdapterMethod | undefined;

      if (nextMethod && isStyleMethod(nextMethod)) {
        onChange({ ...config, method: nextMethod });
      }
    },
    [config, onChange]
  );

  const handleStepsChange = useCallback(
    (beginEndStepPct: [number, number]) => {
      if (beginEndStepPct[0] !== config.beginEndStepPct[0] || beginEndStepPct[1] !== config.beginEndStepPct[1]) {
        onChange({ ...config, beginEndStepPct });
      }
    },
    [config, onChange]
  );

  return (
    <Stack gap="1.5">
      {strengthOnly ? null : (
        <ControlSurface>
          <SegmentGroup.Root
            aria-label={t('widgets.generate.mode')}
            disabled={disabled}
            value={mode}
            w="full"
            onValueChange={handleModeChange}
          >
            <SegmentGroup.Indicator />
            {MODE_SEGMENTS.map((segment) => (
              <SegmentGroup.Item key={segment.value} flex="1" justifyContent="center" value={segment.value}>
                <SegmentGroup.ItemText fontSize="xs">{t(segment.labelKey)}</SegmentGroup.ItemText>
                <SegmentGroup.ItemHiddenInput />
              </SegmentGroup.Item>
            ))}
          </SegmentGroup.Root>
        </ControlSurface>
      )}
      {mode === 'style' && !strengthOnly ? (
        <ControlSurface>
          <Field disabled={disabled} label={t('widgets.generate.styleVariant')} labelPlacement="inside">
            <Select
              collection={styleVariantCollection}
              deselectable={false}
              renderItem={renderStyleVariantItem}
              value={styleVariantValue}
              w="full"
              onValueChange={handleStyleVariantChange}
            />
          </Field>
        </ControlSurface>
      ) : null}
      <ActiveStepsField disabled={disabled} value={config.beginEndStepPct} onChange={handleStepsChange} />
      {children}
    </Stack>
  );
};

/**
 * The step window as one control: its label, a range slider and the window it covers on a single line, the same
 * height and frame as the weight scrubber above it. Every step of a drag commits, so the row's summary follows it.
 */
const ActiveStepsField = ({
  disabled,
  onChange,
  value,
}: {
  disabled: boolean;
  onChange: (value: [number, number]) => void;
  value: [number, number];
}) => {
  const { t } = useTranslation();
  const ariaLabel = useMemo(() => [t('widgets.generate.activeSteps'), t('widgets.generate.activeSteps')], [t]);
  const handleChange = useCallback(
    ({ value: next }: SliderValueChangeDetails) => {
      const begin = next[0];
      const end = next[1];

      if (begin !== undefined && end !== undefined && Number.isFinite(begin) && Number.isFinite(end)) {
        onChange([begin, end]);
      }
    },
    [onChange]
  );

  return (
    <HStack
      aria-disabled={disabled || undefined}
      bg="bg.muted"
      borderColor="border"
      data-control-surface=""
      borderWidth="1px"
      css={disabled ? STEPS_FIELD_DISABLED_CSS : undefined}
      gap="3"
      h="control.md"
      minW="0"
      pe="3"
      ps={INSIDE_LABEL_INSET}
      role="group"
      rounded="control"
    >
      <FeatureHint hint="stepRange">
        <chakra.span flexShrink={0}>
          <FieldLabel>{t('widgets.generate.activeStepsShort')}</FieldLabel>
        </chakra.span>
      </FeatureHint>
      <Slider
        aria-label={ariaLabel}
        disabled={disabled}
        flex="1"
        formatValue={formatPct}
        max={1}
        min={0}
        minW="0"
        step={0.05}
        value={value}
        onValueChange={handleChange}
      />
      <Text flexShrink={0} fontSize="md" fontVariantNumeric="tabular-nums">
        {t('widgets.generate.stepRangeValue', {
          begin: Math.round(value[0] * 100),
          end: Math.round(value[1] * 100),
        })}
      </Text>
    </HStack>
  );
};

export const ClipVisionSelect = ({
  config,
  disabled,
  onChange,
}: {
  config: IPAdapterConfig;
  disabled: boolean;
  onChange: (config: GenerateReferenceImageConfig) => void;
}) => {
  const { t } = useTranslation();
  const [draftClipVisionModel, setDraftClipVisionModel] = useState<ClipVisionModel | null>(null);
  const clipVisionModel = draftClipVisionModel ?? config.clipVisionModel;
  const selectValue = useMemo(() => [clipVisionModel], [clipVisionModel]);

  const handleValueChange = useCallback(
    ({ value }: SelectValueChangeDetails<ClipVisionOption>) => {
      const nextClipVisionModel = value[0];

      if (!isClipVisionModel(nextClipVisionModel)) {
        return;
      }

      setDraftClipVisionModel(nextClipVisionModel);
      globalThis.setTimeout(() => {
        onChange({ ...config, clipVisionModel: nextClipVisionModel });
        setDraftClipVisionModel(null);
      }, 0);
    },
    [config, onChange]
  );

  return (
    <Select
      aria-label={t('widgets.generate.clipVisionModel')}
      collection={CLIP_VISION_COLLECTION}
      deselectable={false}
      disabled={disabled}
      flexShrink="0"
      value={selectValue}
      w="24"
      onValueChange={handleValueChange}
    />
  );
};

export const FluxReduxControls = <T extends { imageInfluence: FluxReduxImageInfluence; type: 'flux_redux' }>({
  config,
  disabled,
  labelPlacement = 'outside',
  onChange,
}: {
  config: T;
  disabled: boolean;
  /** `inside` puts the label in the select's own box, for a single-line row. */
  labelPlacement?: 'inside' | 'outside';
  onChange: (config: T) => void;
}) => {
  const { t } = useTranslation();
  const imageInfluenceValue = useMemo(() => [config.imageInfluence], [config.imageInfluence]);

  const handleValueChange = useCallback(
    ({ value }: SelectValueChangeDetails<FluxReduxImageInfluenceOption>) => {
      const imageInfluence = value[0];

      if (!isFluxReduxImageInfluence(imageInfluence)) {
        return;
      }

      onChange({
        ...config,
        imageInfluence,
      });
    },
    [config, onChange]
  );

  const select = (
    <Select
      collection={FLUX_REDUX_IMAGE_INFLUENCE_COLLECTION}
      deselectable={false}
      disabled={disabled}
      value={imageInfluenceValue}
      w="full"
      onValueChange={handleValueChange}
    />
  );

  return labelPlacement === 'inside' ? (
    <Field
      disabled={disabled}
      hint="imageInfluence"
      label={t('widgets.generate.imageInfluence')}
      labelPlacement="inside"
    >
      {select}
    </Field>
  ) : (
    <Field disabled={disabled} hint="imageInfluence" label={t('widgets.generate.imageInfluence')}>
      {select}
    </Field>
  );
};

const STYLE_STRENGTH_MARKS = [DEFAULT_KREA2_STYLE_STRENGTH];

const KREA2_MODE_SEGMENTS = [
  { labelKey: 'widgets.generate.modeStyle', value: 'style' },
  { labelKey: 'widgets.generate.modeEdit', value: 'edit' },
] as const;

/**
 * Switches a Krea-2 reference between Style and Edit; the image stays and Style restarts at its default strength. It
 * stays live while the row is off, so switched-off references can be turned into the set they will join before they run.
 */
export const Krea2ReferenceModeControl = ({
  config,
  onChange,
}: {
  config: Krea2ReferenceConfig;
  onChange: (config: Krea2ReferenceConfig) => void;
}) => {
  const { t } = useTranslation();
  const mode = config.type === 'krea2_edit_reference_image' ? 'edit' : 'style';
  const handleModeChange = useCallback(
    ({ value }: SegmentGroup.ValueChangeDetails) => {
      if (value === mode) {
        return;
      }

      if (value === 'edit') {
        onChange({ image: config.image, type: 'krea2_edit_reference_image' });
      } else if (value === 'style') {
        onChange({ image: config.image, styleStrength: DEFAULT_KREA2_STYLE_STRENGTH, type: 'krea2_reference_image' });
      }
    },
    [config.image, mode, onChange]
  );

  return (
    <Stack gap="1">
      <FieldHeader hint="krea2ReferenceMode" label={t('widgets.generate.mode')} />
      <ControlSurface>
        <SegmentGroup.Root
          aria-label={t('widgets.generate.mode')}
          value={mode}
          w="full"
          onValueChange={handleModeChange}
        >
          <SegmentGroup.Indicator />
          {KREA2_MODE_SEGMENTS.map((segment) => (
            <SegmentGroup.Item key={segment.value} flex="1" justifyContent="center" value={segment.value}>
              <SegmentGroup.ItemText fontSize="xs">{t(segment.labelKey)}</SegmentGroup.ItemText>
              <SegmentGroup.ItemHiddenInput />
            </SegmentGroup.Item>
          ))}
        </SegmentGroup.Root>
      </ControlSurface>
    </Stack>
  );
};

/** Krea-2 style transfer has one knob; 0 bypasses the reference entirely. */
export const Krea2StyleReferenceControls = <T extends { styleStrength: number; type: 'krea2_reference_image' }>({
  config,
  disabled,
  onChange,
}: {
  config: T;
  disabled: boolean;
  onChange: (config: T) => void;
}) => {
  const { t } = useTranslation();
  const handleStrengthChange = useCallback(
    (styleStrength: number) => onChange({ ...config, styleStrength }),
    [config, onChange]
  );

  return (
    <ScrubberField
      defaultValue={DEFAULT_KREA2_STYLE_STRENGTH}
      disabled={disabled}
      formatValue={formatWeight}
      hint="styleStrength"
      label={t('widgets.generate.styleStrength')}
      marks={STYLE_STRENGTH_MARKS}
      max={MAX_KREA2_STYLE_STRENGTH}
      min={0}
      step={0.05}
      value={config.styleStrength}
      onChange={handleStrengthChange}
    />
  );
};
