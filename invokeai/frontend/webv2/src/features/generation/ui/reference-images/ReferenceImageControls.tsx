import type { SelectValueChangeDetails, SliderValueChangeDetails } from '@chakra-ui/react';
import type {
  ClipVisionModel,
  FluxReduxImageInfluence,
  GenerateReferenceImageConfig,
  IPAdapterMethod,
} from '@features/generation/core/types';
import type { FeatureHintId } from '@platform/ui/hints';
import type { ReactNode } from 'react';

import { Collapsible, createListCollection, HStack, SegmentGroup, Stack, Text } from '@chakra-ui/react';
import { Field, Select, Slider } from '@platform/ui';
import { FeatureHint } from '@platform/ui/hints';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { ChevronRightIcon } from 'lucide-react';
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
    <Text as="span" fontSize="xs">
      {option.label}
    </Text>
    <Text as="span" color="fg.muted" fontSize="2xs" lineHeight="short">
      {option.description}
    </Text>
  </Stack>
);

/** The default weight is the one stop a drag can snap to; the track ends stand in for 0 and 2. */
const WEIGHT_MARKS = [0, 1, 2];
const DEFAULT_WEIGHT = 1;

const BEGIN_END_MARKS = [
  { label: '0%', value: 0 },
  { label: '25%', value: 0.25 },
  { label: '50%', value: 0.5 },
  { label: '75%', value: 0.75 },
  { label: '100%', value: 1 },
];

const ADVANCED_TRIGGER_HOVER_STYLES = { color: 'fg' };
const COLLAPSIBLE_INDICATOR_OPEN_STYLES = { transform: 'rotate(90deg)' };

/** Tiny muted label row above a control, with optional right-aligned content. */
export const FieldHeader = ({
  children,
  hint,
  label,
}: {
  children?: ReactNode;
  hint?: FeatureHintId;
  label: string;
}) => {
  const labelText = (
    <Text color="fg.muted" fontSize="2xs" fontWeight="medium">
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

/** Render children inside Advanced. */
export const IPAdapterControls = ({
  children,
  config,
  disabled,
  onChange,
}: {
  children?: ReactNode;
  config: IPAdapterConfig;
  disabled: boolean;
  onChange: (config: GenerateReferenceImageConfig) => void;
}) => {
  const { t } = useTranslation();
  const [draftBeginEndStepPct, setDraftBeginEndStepPct] = useState<[number, number] | null>(null);
  const mode = getReferenceMode(config.method);
  const beginEndStepPct = draftBeginEndStepPct ?? config.beginEndStepPct;
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

  // Do not compare against captured drag-start values; returning to the start still needs a commit.
  const handleWeightChange = useCallback((weight: number) => onChange({ ...config, weight }), [config, onChange]);

  const commitBeginEndStepPct = useCallback(
    (nextBeginEndStepPct: [number, number]) => {
      setDraftBeginEndStepPct(null);

      if (
        nextBeginEndStepPct[0] !== config.beginEndStepPct[0] ||
        nextBeginEndStepPct[1] !== config.beginEndStepPct[1]
      ) {
        onChange({ ...config, beginEndStepPct: nextBeginEndStepPct });
      }
    },
    [config, onChange]
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

  const handleStepsChange = useCallback(({ value }: SliderValueChangeDetails) => {
    const begin = value[0];
    const end = value[1];

    if (Number.isFinite(begin) && Number.isFinite(end)) {
      setDraftBeginEndStepPct([begin as number, end as number]);
    }
  }, []);

  const handleStepsChangeEnd = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const begin = value[0];
      const end = value[1];

      if (Number.isFinite(begin) && Number.isFinite(end)) {
        commitBeginEndStepPct([begin as number, end as number]);
      }
    },
    [commitBeginEndStepPct]
  );

  const stepsAriaLabel = useMemo(() => [t('widgets.generate.activeSteps'), t('widgets.generate.activeSteps')], [t]);

  return (
    <Stack gap="2">
      <Stack gap="1">
        <FieldHeader label={t('widgets.generate.mode')} />
        <SegmentGroup.Root disabled={disabled} size="xs" value={mode} w="full" onValueChange={handleModeChange}>
          <SegmentGroup.Indicator />
          {MODE_SEGMENTS.map((segment) => (
            <SegmentGroup.Item key={segment.value} flex="1" justifyContent="center" value={segment.value}>
              <SegmentGroup.ItemText fontSize="2xs">{t(segment.labelKey)}</SegmentGroup.ItemText>
              <SegmentGroup.ItemHiddenInput />
            </SegmentGroup.Item>
          ))}
        </SegmentGroup.Root>
      </Stack>

      <ScrubberField
        defaultValue={DEFAULT_WEIGHT}
        disabled={disabled}
        formatValue={formatWeight}
        hint="referenceImageWeight"
        inputMin={-1}
        label={t('widgets.generate.weight')}
        marks={WEIGHT_MARKS}
        max={2}
        min={0}
        step={0.05}
        value={config.weight}
        onChange={handleWeightChange}
      />

      <Collapsible.Root>
        <Collapsible.Trigger
          alignItems="center"
          color="fg.muted"
          display="flex"
          fontSize="2xs"
          fontWeight="medium"
          gap="1"
          _hover={ADVANCED_TRIGGER_HOVER_STYLES}
        >
          <Collapsible.Indicator
            _open={COLLAPSIBLE_INDICATOR_OPEN_STYLES}
            transition="transform var(--wb-motion-duration-slow)"
          >
            <ChevronRightIcon size="12" />
          </Collapsible.Indicator>
          {t('widgets.generate.advanced')}
        </Collapsible.Trigger>
        <Collapsible.Content>
          <Stack borderTopWidth="1px" gap="2" mt="2" pt="2">
            {children}
            {mode === 'style' ? (
              <Stack gap="1">
                <FieldHeader label={t('widgets.generate.styleVariant')} />
                <Select
                  collection={styleVariantCollection}
                  deselectable={false}
                  disabled={disabled}
                  renderItem={renderStyleVariantItem}
                  size="xs"
                  value={styleVariantValue}
                  w="full"
                  onValueChange={handleStyleVariantChange}
                />
              </Stack>
            ) : null}
            <Stack gap="1">
              <FieldHeader label={t('widgets.generate.activeSteps')}>
                <Text color="fg.subtle" fontFamily="mono" fontSize="2xs">
                  {formatPct(beginEndStepPct[0])} – {formatPct(beginEndStepPct[1])}
                </Text>
              </FieldHeader>
              <Slider
                aria-label={stepsAriaLabel}
                disabled={disabled}
                formatValue={formatPct}
                marks={BEGIN_END_MARKS}
                max={1}
                min={0}
                size="sm"
                step={0.05}
                value={beginEndStepPct}
                onValueChange={handleStepsChange}
                onValueChangeEnd={handleStepsChangeEnd}
              />
            </Stack>
          </Stack>
        </Collapsible.Content>
      </Collapsible.Root>
    </Stack>
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
      collection={CLIP_VISION_COLLECTION}
      deselectable={false}
      disabled={disabled}
      flexShrink="0"
      size="xs"
      value={selectValue}
      w="24"
      onValueChange={handleValueChange}
    />
  );
};

export const FluxReduxControls = <T extends { imageInfluence: FluxReduxImageInfluence; type: 'flux_redux' }>({
  config,
  disabled,
  onChange,
}: {
  config: T;
  disabled: boolean;
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

  return (
    <Field disabled={disabled} hint="imageInfluence" label={t('widgets.generate.imageInfluence')}>
      <Select
        collection={FLUX_REDUX_IMAGE_INFLUENCE_COLLECTION}
        deselectable={false}
        disabled={disabled}
        size="xs"
        value={imageInfluenceValue}
        w="full"
        onValueChange={handleValueChange}
      />
    </Field>
  );
};
