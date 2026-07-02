import type { SelectValueChangeDetails, SliderValueChangeDetails } from '@chakra-ui/react';
import type {
  ClipVisionModel,
  FluxReduxImageInfluence,
  GenerateReferenceImageConfig,
  IPAdapterMethod,
} from '@workbench/generation/types';
import type { KeyboardEvent, ReactNode } from 'react';

import {
  Collapsible,
  createListCollection,
  Flex,
  HStack,
  NumberInput,
  SegmentGroup,
  Stack,
  Text,
} from '@chakra-ui/react';
import { Field, Select, Slider } from '@workbench/components/ui';
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

const WEIGHT_MARKS = [
  { label: '0', value: 0 },
  { label: '1', value: 1 },
  { label: '2', value: 2 },
];

const BEGIN_END_MARKS = [
  { label: '0%', value: 0 },
  { label: '25%', value: 0.25 },
  { label: '50%', value: 0.5 },
  { label: '75%', value: 0.75 },
  { label: '100%', value: 1 },
];

const SEGMENT_CHECKED_STYLES = { color: 'accent.contrast' };
const ADVANCED_TRIGGER_HOVER_STYLES = { color: 'fg' };
const COLLAPSIBLE_INDICATOR_OPEN_STYLES = { transform: 'rotate(90deg)' };

/** Tiny muted label row above a control, with optional right-aligned content. */
export const FieldHeader = ({ children, label }: { children?: ReactNode; label: string }) => (
  <HStack justify="space-between" minH="4">
    <Text color="fg.muted" fontSize="2xs" fontWeight="medium">
      {label}
    </Text>
    {children}
  </HStack>
);

/**
 * Controls for `ip_adapter` reference image configs. `children` renders inside
 * the Advanced section (the card passes the model selector through here).
 */
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
  const [draftWeight, setDraftWeight] = useState<number | null>(null);
  const [draftBeginEndStepPct, setDraftBeginEndStepPct] = useState<[number, number] | null>(null);
  const mode = getReferenceMode(config.method);
  const weight = draftWeight ?? config.weight;
  const sliderWeight = Math.max(0, weight);
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

  const commitWeight = useCallback(
    (nextWeight: number) => {
      setDraftWeight(null);

      if (nextWeight !== config.weight) {
        onChange({ ...config, weight: nextWeight });
      }
    },
    [config, onChange]
  );

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

  const handleWeightInputChange = useCallback(({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
    if (Number.isFinite(valueAsNumber)) {
      setDraftWeight(valueAsNumber);
    }
  }, []);

  const handleWeightInputBlur = useCallback(() => commitWeight(weight), [commitWeight, weight]);

  const handleWeightInputKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') {
        commitWeight(weight);
      }
    },
    [commitWeight, weight]
  );

  const handleWeightSliderChange = useCallback(({ value }: SliderValueChangeDetails) => {
    if (Number.isFinite(value[0])) {
      setDraftWeight(value[0] as number);
    }
  }, []);

  const handleWeightSliderChangeEnd = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      if (Number.isFinite(value[0])) {
        commitWeight(value[0] as number);
      }
    },
    [commitWeight]
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

  const weightSliderValue = useMemo(() => [sliderWeight], [sliderWeight]);
  const weightAriaLabel = useMemo(() => [t('widgets.generate.weight')], [t]);
  const stepsAriaLabel = useMemo(() => [t('widgets.generate.activeSteps'), t('widgets.generate.activeSteps')], [t]);

  return (
    <Stack gap="2">
      <Stack gap="1">
        <FieldHeader label={t('widgets.generate.mode')} />
        <SegmentGroup.Root disabled={disabled} size="xs" value={mode} w="full" onValueChange={handleModeChange}>
          <SegmentGroup.Indicator bg="accent.solid" />
          {MODE_SEGMENTS.map((segment) => (
            <SegmentGroup.Item
              key={segment.value}
              flex="1"
              justifyContent="center"
              value={segment.value}
              _checked={SEGMENT_CHECKED_STYLES}
            >
              <SegmentGroup.ItemText fontSize="2xs">{t(segment.labelKey)}</SegmentGroup.ItemText>
              <SegmentGroup.ItemHiddenInput />
            </SegmentGroup.Item>
          ))}
        </SegmentGroup.Root>
      </Stack>

      <Field align="center" disabled={disabled} gap="3" label={t('widgets.generate.weight')} orientation="horizontal">
        <Flex align="center" gap="3">
          <Slider
            aria-label={weightAriaLabel}
            formatValue={formatWeight}
            marks={WEIGHT_MARKS}
            max={2}
            min={0}
            disabled={disabled}
            size="sm"
            step={0.05}
            value={weightSliderValue}
            onValueChange={handleWeightSliderChange}
            onValueChangeEnd={handleWeightSliderChangeEnd}
            w="full"
          />

          <NumberInput.Root
            disabled={disabled}
            max={2}
            min={-1}
            size="xs"
            step={0.05}
            value={String(weight)}
            w="20"
            onValueChange={handleWeightInputChange}
          >
            <NumberInput.Control />
            <NumberInput.Input
              aria-label={t('widgets.generate.weight')}
              fontSize="xs"
              onBlur={handleWeightInputBlur}
              onKeyDown={handleWeightInputKeyDown}
            />
          </NumberInput.Root>
        </Flex>
      </Field>

      <Collapsible.Root>
        <Collapsible.Trigger
          alignItems="center"
          color="fg.muted"
          cursor="pointer"
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

export const FluxReduxControls = ({
  config,
  disabled,
  onChange,
}: {
  config: Extract<GenerateReferenceImageConfig, { type: 'flux_redux' }>;
  disabled: boolean;
  onChange: (config: GenerateReferenceImageConfig) => void;
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
    <Field disabled={disabled} label={t('widgets.generate.imageInfluence')}>
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
