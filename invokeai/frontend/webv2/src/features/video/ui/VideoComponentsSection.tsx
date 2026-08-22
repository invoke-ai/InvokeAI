import type { MainModelConfig, ModelIdentifierConfig, VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { VideoWidgetValues } from '@features/video/core/types';
import type {
  VideoComponentPolicyContext,
  VideoComponentSlotPolicy,
  VideoComponentValueKey,
} from '@features/video/core/videoPolicies';

import { Stack } from '@chakra-ui/react';
import { GenerationSettingsSection } from '@features/generation/components';
import { isMainModelConfig, isModelIdentifierConfig, isVaeModelConfig } from '@features/generation/settings';
import { ModelSelect } from '@features/models/react';
import { getVideoComponentSectionPolicy } from '@features/video/core/videoPolicies';
import { Field } from '@platform/ui';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Model Components for the Video panel — fully policy-driven, like the
 * generation panel's section: the component renders whatever slots the
 * capability matrix returns (Wan: component source / VAE / Wan T5 / low-noise
 * expert; MiniMax H3: the two single-file overrides), with each slot's
 * compatibility filter and requiredness coming from the policy.
 */

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
      helpText={isMissing ? undefined : slot.helpText}
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
          <ComponentSlotRow
            key={slot.key}
            ctx={ctx}
            slot={slot}
            value={values[slot.key as VideoComponentValueKey]}
            onPatch={onPatch}
          />
        ))}
      </Stack>
    </GenerationSettingsSection>
  );
});
