import type { MainModelConfig, ModelIdentifierConfig, VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { VideoWidgetValues } from '@features/video/core/types';
import type {
  VideoComponentPolicyContext,
  VideoComponentSlotPolicy,
  VideoComponentValueKey,
} from '@features/video/core/videoPolicies';

import { HStack, Stack, Text } from '@chakra-ui/react';
import { GenerationSettingsSection } from '@features/generation/components';
import { isMainModelConfig, isModelIdentifierConfig, isVaeModelConfig } from '@features/generation/settings';
import { useModelsSelector } from '@features/models';
import { ModelSelect } from '@features/models/react';
import { MINIMAX_H3_HYBRID_BLOCK_RANGE } from '@features/video/core/settings';
import {
  getVideoComponentSectionPolicy,
  getVideoModelSelectionResult,
  getWanExpertWiringWarning,
} from '@features/video/core/videoPolicies';
import { Field } from '@platform/ui';
import { Button } from '@platform/ui/Button';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Fragment, memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const coerceSlotValue = (
  slot: VideoComponentSlotPolicy,
  candidate: ModelConfig | null
): MainModelConfig | VaeModelConfig | ModelIdentifierConfig | null => {
  if (!candidate) {
    return null;
  }

  switch (slot.valueKind) {
    case 'main':
      return isMainModelConfig(candidate) ? candidate : null;
    case 'vae':
      return isVaeModelConfig(candidate) ? candidate : null;
    case 'component':
      return isModelIdentifierConfig(candidate) ? candidate : null;
  }
};

const ComponentSlotRow = memo(function ComponentSlotRow({
  ctx,
  onPatch,
  slot,
  value,
}: {
  ctx: VideoComponentPolicyContext;
  onPatch: (patch: Partial<VideoWidgetValues>) => void;
  slot: VideoComponentSlotPolicy;
  value: ModelIdentifierConfig | MainModelConfig | VaeModelConfig | null;
}) {
  const { t } = useTranslation();
  const isMissing = Boolean(slot.required?.(ctx)) && !value;
  const filter = useMemo(
    () => (slot.filter ? (candidate: ModelConfig) => slot.filter?.(candidate, ctx) ?? true : undefined),
    [ctx, slot]
  );
  const handleChange = useMemo(
    () => (candidate: ModelConfig | null) => onPatch({ [slot.key]: coerceSlotValue(slot, candidate) }),
    [onPatch, slot]
  );

  return (
    <Field
      error={isMissing ? (slot.missingMessage ?? t('widgets.video.componentRequired')) : undefined}
      helpText={isMissing || !slot.helpTextKey ? undefined : t(slot.helpTextKey)}
      label={slot.label}
    >
      <ModelSelect
        filter={filter}
        invalid={isMissing}
        isClearable
        modelTypes={slot.modelTypes}
        placeholder={t('widgets.video.selectComponent')}
        size="xs"
        value={value?.key ?? null}
        onChange={handleChange}
      />
    </Field>
  );
});

/** Blocks at and above this index use Ref2VA AdaLN projections; earlier blocks use the FL2VA base. */
const HybridStartBlockRow = memo(function HybridStartBlockRow({
  onPatch,
  value,
}: {
  onPatch: (patch: Partial<VideoWidgetValues>) => void;
  value: number;
}) {
  const { t } = useTranslation();
  const handleChange = useCallback((h3HybridStartBlock: number) => onPatch({ h3HybridStartBlock }), [onPatch]);

  // Reuse ScrubberField to avoid adding a shared Generate-only control chunk to editor boot.
  return (
    <ScrubberField
      defaultValue={MINIMAX_H3_HYBRID_BLOCK_RANGE.defaultStart}
      helpText={t('widgets.video.hybridStartBlockHelp')}
      label={t('widgets.video.hybridStartBlock')}
      max={MINIMAX_H3_HYBRID_BLOCK_RANGE.max}
      min={MINIMAX_H3_HYBRID_BLOCK_RANGE.min}
      step={1}
      value={value}
      onChange={handleChange}
    />
  );
});

/**
 * Expert tags are filename heuristics; explicit wiring remains authoritative, so warn and offer a swap without
 * blocking.
 */
// Spelled out rather than interpolated, so the translation-key scan can see
// them and fail the build if a string goes missing.
const EXPERT_WIRING_MESSAGE_KEYS = {
  'high-as-low': 'widgets.video.expertWiring.highAsLow',
  'low-as-main': 'widgets.video.expertWiring.lowAsMain',
  'single-low': 'widgets.video.expertWiring.singleLow',
  swapped: 'widgets.video.expertWiring.swapped',
} as const;

const WanExpertWiringNotice = memo(function WanExpertWiringNotice({
  onPatch,
  values,
}: {
  onPatch: (patch: Partial<VideoWidgetValues>) => void;
  values: VideoWidgetValues;
}) {
  const { t } = useTranslation();
  const models = useModelsSelector((snapshot) => snapshot.models);
  // Wait for catalog authority before judging the accelerator pair installed.
  const modelsLoaded = useModelsSelector((snapshot) => snapshot.status) === 'loaded';
  const warning = useMemo(
    () => getWanExpertWiringWarning(values.model, values.wanLowNoiseModel),
    [values.model, values.wanLowNoiseModel]
  );
  // Offer only swaps that clear the warning and whose two configs still exist.
  const bothExpertsInstalled =
    Boolean(values.model && models.some((candidate) => candidate.key === values.model?.key)) &&
    Boolean(values.wanLowNoiseModel && models.some((candidate) => candidate.key === values.wanLowNoiseModel?.key));
  const swapResolves =
    warning !== null &&
    warning.kind !== 'single-low' &&
    bothExpertsInstalled &&
    values.wanLowNoiseModel?.format !== 'diffusers' &&
    getWanExpertWiringWarning(values.wanLowNoiseModel, values.model) === null;
  const swapExperts = useCallback(() => {
    const previousMain = values.model;
    const nextMain = values.wanLowNoiseModel;

    if (!previousMain || !nextMain || !models.some((candidate) => candidate.key === nextMain.key)) {
      return;
    }

    const result = getVideoModelSelectionResult({ currentSettings: values, model: nextMain, models });

    onPatch({ ...result.settings, model: nextMain, wanLowNoiseModel: previousMain });
  }, [models, onPatch, values]);

  if (!warning) {
    return null;
  }

  const message = t(EXPERT_WIRING_MESSAGE_KEYS[warning.kind], {
    lowName: values.wanLowNoiseModel?.name ?? '',
    mainName: values.model?.name ?? '',
  });

  return (
    <HStack bg="bg.subtle" borderColor="fg.warning" borderWidth="1px" gap="2" p="2" rounded="md">
      <Text color="fg.warning" flex="1" fontSize="2xs" textWrap="pretty">
        {message}
      </Text>
      {swapResolves && modelsLoaded ? (
        <Button flexShrink="0" size="2xs" variant="outline" onClick={swapExperts}>
          {t('widgets.video.expertWiring.swap')}
        </Button>
      ) : null}
    </HStack>
  );
});

export const VideoComponentsSection = memo(function VideoComponentsSection({
  onPatch,
  values,
}: {
  onPatch: (patch: Partial<VideoWidgetValues>) => void;
  values: VideoWidgetValues;
}) {
  const { t } = useTranslation();
  const policy = useMemo(() => getVideoComponentSectionPolicy(values.model ?? undefined, values), [values]);
  const ctx = useMemo<VideoComponentPolicyContext | null>(
    () => (values.model ? { model: values.model, selectedComponents: values, settings: values } : null),
    [values]
  );

  if (!ctx || policy.slots.length === 0) {
    return null;
  }

  return (
    <GenerationSettingsSection
      defaultOpen={policy.defaultOpen}
      label={t('widgets.video.components')}
      sectionId="video-components"
    >
      <Stack gap="3" p="2">
        {policy.slots.map((slot) => (
          <Fragment key={slot.key}>
            <ComponentSlotRow
              ctx={ctx}
              slot={slot}
              value={values[slot.key as VideoComponentValueKey]}
              onPatch={onPatch}
            />
            {slot.key === 'h3HybridBaseModel' && values.h3HybridBaseModel ? (
              <HybridStartBlockRow value={values.h3HybridStartBlock} onPatch={onPatch} />
            ) : null}
          </Fragment>
        ))}
        <WanExpertWiringNotice values={values} onPatch={onPatch} />
      </Stack>
    </GenerationSettingsSection>
  );
});
