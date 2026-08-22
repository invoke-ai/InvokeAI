import type { ModelConfig, ModelTaxonomyType } from '@features/models';
import type { VideoWidgetValues } from '@features/video/core/types';

import { createListCollection, HStack, NumberInput, Stack, Switch, Text } from '@chakra-ui/react';
import { GenerationSettingsSection } from '@features/generation/components';
import { isMainModelConfig, sanitizeBatchCount, SEED_MAX } from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { ModelSelect } from '@features/models/react';
import { getVideoDurationSeconds, invertVideoAspectRatioId } from '@features/video/core/dimensions';
import { normalizeVideoWidgetValues, VIDEO_ASPECT_RATIO_IDS } from '@features/video/core/settings';
import {
  getAcceleratorToggleResult,
  getVideoDimensions,
  getVideoModelPolicy,
  getVideoModelSelectionResult,
  isVideoModelSelectable,
} from '@features/video/core/videoPolicies';
import { createDefaultVideoWidgetValues, syncVideoWidgetValuesWithModels } from '@features/video/core/widgetValues';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Field, IconButton, Select } from '@platform/ui';
import { SliderNumberField } from '@platform/ui/SliderNumberField';
import { toaster } from '@platform/ui/toaster';
import { ArrowLeftRightIcon, DicesIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { areVideoValuesEqual } from './videoComparators';
import { VideoPromptFields } from './VideoFormFields';
import { useVideoUi, useVideoUiActions } from './VideoUiContext';

/**
 * Every prop identity in this file is stable by construction — module-scope
 * constants for literals, `useCallback`/`useMemo` for anything closing over
 * state, and `memo` on each section — matching the Upscale widget's contract:
 * the widget re-renders on every keystroke that patches project state.
 */

const MAIN_MODEL_TYPES: readonly ModelTaxonomyType[] = ['main'];
const SWITCH_CHECKED_PROPS = { bg: 'accent.solid' };

const ASPECT_RATIO_COLLECTION = createListCollection({
  items: VIDEO_ASPECT_RATIO_IDS.map((id) => ({ label: id, value: id })),
});

const toTargetResolution = (value: string | undefined): VideoWidgetValues['targetResolution'] | null =>
  value === '480p' || value === '720p' || value === '1080p' || value === '768 highres' || value === '768 lowres'
    ? value
    : null;

const DURATION_FORMATTER = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 2,
  minimumFractionDigits: 1,
});

const VideoModelReconciler = ({
  rawValues,
  values,
}: {
  rawValues: Record<string, unknown>;
  values: VideoWidgetValues;
}) => {
  const { patchValues } = useVideoUiActions();

  useMountEffect(() => {
    const normalized = normalizeVideoWidgetValues(rawValues);

    if (normalized && areVideoValuesEqual(normalized, values)) {
      return;
    }

    // When the store fails normalization wholesale (first open: the topbar
    // Iterations field may already have patched `batchCount` into an
    // otherwise-empty widget store), seeding the defaults must not wipe that
    // one pre-open edit.
    const batchCount = normalized ? values.batchCount : sanitizeBatchCount(rawValues.batchCount ?? values.batchCount);

    patchValues({ ...values, batchCount }, 'system');
  });

  return null;
};

export const VideoWidgetView = () => {
  const { t } = useTranslation();
  const selection = useVideoUi();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const { patchPromptDraft: patchDraft, patchValues, projectId, promptDraft, rawValues } = selection;
  // Normalizing and reconciling against the model list is the widget's most
  // expensive derivation; it must not run on unrelated re-renders, and a fresh
  // `values` identity would re-render every section below.
  const values = useMemo(() => {
    const normalized =
      normalizeVideoWidgetValues(rawValues) ?? createDefaultVideoWidgetValues(modelsStatus === 'loaded' ? models : []);

    return modelsStatus === 'loaded' ? syncVideoWidgetValuesWithModels(normalized, models) : normalized;
  }, [models, modelsStatus, rawValues]);
  const modelsFingerprint = useMemo(
    () =>
      models
        .map(
          (model) =>
            `${model.key}:${model.hash}:${model.name}:${model.base}:${model.type}:${model.format}:${model.variant ?? ''}`
        )
        .join('|'),
    [models]
  );
  const policy = useMemo(() => getVideoModelPolicy(values.model ?? undefined, values), [values]);
  const dimensions = useMemo(() => getVideoDimensions(values.model ?? undefined, values), [values]);
  const durationSeconds = getVideoDurationSeconds(
    values.numFrames,
    policy.fps.editable ? values.fps : policy.fps.defaultValue
  );

  const patch = useCallback((next: Partial<VideoWidgetValues>) => patchValues(next), [patchValues]);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const selectMainModel = useCallback(
    (model: ModelConfig | null) => {
      if (!isMainModelConfig(model) || !isVideoModelSelectable(model)) {
        return;
      }

      const result = getVideoModelSelectionResult({ currentSettings: values, model, models });

      patch({ ...result.settings, model });

      if (result.clearedLabels.length > 0) {
        toaster.create({
          description: t('widgets.video.settingsAdjustedDescription', {
            labels: result.clearedLabels.join(', '),
          }),
          title: t('widgets.video.settingsAdjusted'),
          type: 'info',
        });
      }
    },
    [models, patch, t, values]
  );

  const toggleAccelerator = useCallback(
    (details: { checked: boolean }) => {
      if (!values.model) {
        return;
      }

      const result = getAcceleratorToggleResult(values, values.model, models, details.checked);

      if (result.missingLoras) {
        toaster.create({
          description: t('widgets.video.acceleratorMissingDescription', {
            label: policy.ui.accelerator?.label ?? '',
          }),
          title: t('widgets.video.acceleratorMissing'),
          type: 'warning',
        });
        return;
      }

      patch({ ...result.settings });
    },
    [models, patch, policy.ui.accelerator?.label, t, values]
  );

  // One setter per field, created once per `patch` identity: inline
  // `onChange={(x) => patch({ x })}` props would defeat every `memo` below.
  const set = useMemo(
    () => ({
      aspectRatio: ({ value }: { value: string[] }) => {
        const aspectRatioId = value[0];

        if (aspectRatioId && (VIDEO_ASPECT_RATIO_IDS as readonly string[]).includes(aspectRatioId)) {
          patch({ aspectRatioId: aspectRatioId as VideoWidgetValues['aspectRatioId'] });
        }
      },
      cfgScale: (cfgScale: number) => patch({ cfgScale }),
      cfgScaleLowNoise: (cfgScaleLowNoise: number) => patch({ cfgScaleLowNoise }),
      fps: (fps: number) => patch({ fps }),
      numFrames: (numFrames: number) => patch({ numFrames }),
      randomizeSeed: (details: { checked: boolean }) => patch({ shouldRandomizeSeed: details.checked }),
      seed: ({ valueAsNumber }: NumberInput.ValueChangeDetails) =>
        Number.isFinite(valueAsNumber) && patch({ seed: valueAsNumber }),
      shuffleSeed: () => patch({ seed: Math.floor(Math.random() * SEED_MAX) }),
      steps: (steps: number) => patch({ steps }),
      targetResolution: ({ value }: { value: string[] }) => {
        const targetResolution = toTargetResolution(value[0]);

        if (targetResolution) {
          patch({ targetResolution });
        }
      },
    }),
    [patch]
  );

  const swapAspectRatio = useCallback(
    () => patch({ aspectRatioId: invertVideoAspectRatioId(values.aspectRatioId) }),
    [patch, values.aspectRatioId]
  );

  const targetResolutionCollection = useMemo(
    () => createListCollection({ items: policy.targetResolutions.map((option) => ({ ...option, value: option.id })) }),
    [policy.targetResolutions]
  );
  const aspectRatioValue = useMemo(() => [values.aspectRatioId], [values.aspectRatioId]);
  const targetResolutionValue = useMemo(() => [values.targetResolution], [values.targetResolution]);

  const framesSlider = useMemo(
    () =>
      policy.frames.kind === 'grid'
        ? { max: policy.frames.max, min: policy.frames.min, step: policy.frames.step }
        : {
            max: policy.frames.choices[policy.frames.choices.length - 1] ?? 0,
            min: policy.frames.choices[0] ?? 0,
            step:
              policy.frames.choices.length > 1 ? (policy.frames.choices[1] ?? 0) - (policy.frames.choices[0] ?? 0) : 1,
          },
    [policy.frames]
  );

  // Conditioning media locks the aspect ratio in later stack PRs; today the
  // preset is always the source, so this only reports the derived canvas.
  const derivedSizeText = dimensions
    ? t('widgets.video.derivedSize', { height: dimensions.height, width: dimensions.width })
    : t('widgets.video.derivedSizeUnavailable');
  const durationText =
    durationSeconds === null
      ? undefined
      : t('widgets.video.framesDuration', { seconds: DURATION_FORMATTER.format(durationSeconds) });

  return (
    <Stack gap="1" minW="0" p="1">
      <VideoModelReconciler
        key={`${projectId}:${modelsStatus}:${modelsFingerprint}`}
        rawValues={rawValues}
        values={values}
      />

      <VideoPromptFields
        loras={values.loras}
        model={values.model}
        negativeHelpText={policy.prompt.negativeHelpText}
        negativePromptHeightPx={values.negativePromptHeightPx}
        negativeVisible={policy.prompt.negativeVisible}
        positivePromptHeightPx={values.positivePromptHeightPx}
        projectId={projectId}
        promptDraft={promptDraft}
        showSyntaxHighlighting={selection.showPromptSyntaxHighlighting}
        onPatchPromptDraft={patchDraft}
        onPatchValues={patch}
      />

      <GenerationSettingsSection label={t('widgets.video.dimensions')} sectionId="video-dimensions" defaultOpen>
        <Stack gap="3" p="2">
          <Field helpText={derivedSizeText} label={t('widgets.video.aspectRatio')}>
            <HStack gap="1">
              <Select
                collection={ASPECT_RATIO_COLLECTION}
                flex="1"
                size="xs"
                value={aspectRatioValue}
                onValueChange={set.aspectRatio}
              />
              <IconButton
                aria-label={t('widgets.video.swapAspectRatio')}
                size="xs"
                variant="ghost"
                onClick={swapAspectRatio}
              >
                <ArrowLeftRightIcon />
              </IconButton>
            </HStack>
          </Field>
          <Field label={t('widgets.video.targetResolution')}>
            <Select
              collection={targetResolutionCollection}
              size="xs"
              value={targetResolutionValue}
              onValueChange={set.targetResolution}
            />
          </Field>
          <Field helpText={durationText} label={t('widgets.video.frames')}>
            <SliderNumberField
              ariaLabel={t('widgets.video.frames')}
              max={framesSlider.max}
              min={framesSlider.min}
              step={framesSlider.step}
              value={values.numFrames}
              onChange={set.numFrames}
            />
          </Field>
          {policy.ui.fpsVisible ? (
            <Field label={t('widgets.video.fps')}>
              <SliderNumberField
                ariaLabel={t('widgets.video.fps')}
                max={60}
                min={policy.fps.min}
                numberInputMax={policy.fps.max}
                step={1}
                value={values.fps}
                onChange={set.fps}
              />
            </Field>
          ) : (
            <Text color="fg.muted" fontSize="2xs">
              {t('widgets.video.fixedFps', { fps: policy.fps.defaultValue })}
            </Text>
          )}
        </Stack>
      </GenerationSettingsSection>

      <GenerationSettingsSection label={t('widgets.video.model')} sectionId="video-model" defaultOpen>
        <Stack gap="3" p="2">
          <Field
            error={values.model ? undefined : t('widgets.video.modelRequired')}
            label={t('widgets.video.mainModel')}
          >
            <ModelSelect
              filter={isVideoModelSelectable}
              invalid={!values.model}
              modelTypes={MAIN_MODEL_TYPES}
              placeholder={t('widgets.video.selectModel')}
              size="xs"
              value={values.model?.key ?? null}
              onChange={selectMainModel}
            />
          </Field>
          {policy.ui.accelerator && values.model ? (
            <Field
              helpText={t('widgets.video.acceleratorHelp', {
                label: policy.ui.accelerator.label,
                steps: policy.ui.accelerator.steps,
              })}
              label={t('widgets.video.accelerator', { label: policy.ui.accelerator.label })}
            >
              <Switch.Root checked={values.acceleratorEnabled} size="sm" onCheckedChange={toggleAccelerator}>
                <Switch.HiddenInput />
                <Switch.Control _checked={SWITCH_CHECKED_PROPS}>
                  <Switch.Thumb />
                </Switch.Control>
              </Switch.Root>
            </Field>
          ) : null}
          <Field hint="steps" label={t('widgets.video.steps')}>
            <SliderNumberField
              ariaLabel={t('widgets.video.steps')}
              max={100}
              min={policy.minSteps}
              numberInputMax={500}
              step={1}
              value={values.steps}
              onChange={set.steps}
            />
          </Field>
          {policy.ui.cfgVisible ? (
            <Field hint="cfgScale" label={t('widgets.video.cfg')}>
              <SliderNumberField
                ariaLabel={t('widgets.video.cfg')}
                max={15}
                min={1}
                numberInputMax={100}
                step={0.1}
                value={values.cfgScale}
                onChange={set.cfgScale}
              />
            </Field>
          ) : null}
          {policy.ui.cfgLowNoiseVisible ? (
            <Field helpText={t('widgets.video.cfgLowNoiseHelp')} label={t('widgets.video.cfgLowNoise')}>
              <SliderNumberField
                ariaLabel={t('widgets.video.cfgLowNoise')}
                max={15}
                min={0}
                numberInputMax={100}
                step={0.1}
                value={values.cfgScaleLowNoise ?? values.cfgScale}
                onChange={set.cfgScaleLowNoise}
              />
            </Field>
          ) : null}
          <Field hint="seed" label={t('widgets.video.seed')}>
            <HStack gap="2">
              <NumberInput.Root
                disabled={values.shouldRandomizeSeed}
                flex="1"
                max={SEED_MAX}
                min={0}
                size="xs"
                step={1}
                value={String(values.seed)}
                onValueChange={set.seed}
              >
                <NumberInput.Input aria-label={t('widgets.video.seed')} />
              </NumberInput.Root>
              <IconButton
                aria-label={t('widgets.video.shuffleSeed')}
                disabled={values.shouldRandomizeSeed}
                size="xs"
                variant="ghost"
                onClick={set.shuffleSeed}
              >
                <DicesIcon />
              </IconButton>
              <HStack gap="1">
                <Switch.Root checked={values.shouldRandomizeSeed} size="sm" onCheckedChange={set.randomizeSeed}>
                  <Switch.HiddenInput />
                  <Switch.Control _checked={SWITCH_CHECKED_PROPS}>
                    <Switch.Thumb />
                  </Switch.Control>
                </Switch.Root>
                <Text color="fg.muted" fontSize="2xs">
                  {t('widgets.video.randomizeSeed')}
                </Text>
              </HStack>
            </HStack>
          </Field>
        </Stack>
      </GenerationSettingsSection>
    </Stack>
  );
};
