import type {
  VideoConditioningClip,
  VideoReferenceItem,
  VideoSourceClip,
  VideoWidgetValues,
} from '@features/video/core/types';

import { createListCollection, HStack, Stack, Switch, Text } from '@chakra-ui/react';
import { GenerationSettingsSection, SeedField } from '@features/generation/components';
import { sanitizeBatchCount } from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import {
  getVideoDurationSeconds,
  invertVideoAspectRatioId,
  LTX2_EXTEND_CONTEXT_FRAMES,
  LTX2_NUM_FRAMES_STEP,
  snapLtx2FramesDown,
} from '@features/video/core/dimensions';
import {
  applyReferenceExtendNumFrames,
  canPlaceReferenceExtendAnchor,
  createFrameImageSetter,
  getConditioningClipPatch,
  getInitialVideoPatch,
  getReferencesPatch,
  isConditioningClipExcludingFrames,
  isVideoTargetResolution,
  normalizeVideoWidgetValues,
  resolveVideoMode,
  VIDEO_ASPECT_RATIO_IDS,
} from '@features/video/core/settings';
import {
  getAcceleratorLoraChangeResult,
  getAcceleratorToggleResult,
  getEffectiveVideoTiming,
  getVideoDimensions,
  getAutoDurationBounds,
  getVideoExpandPromptSuggestion,
  getVideoModelPolicy,
  isAutoDurationActive,
  isAutoDurationSupportedForMode,
} from '@features/video/core/videoPolicies';
import { createDefaultVideoWidgetValues, syncVideoWidgetValuesWithModels } from '@features/video/core/widgetValues';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Field, IconButton, Select } from '@platform/ui';
import { Button } from '@platform/ui/Button';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { toaster } from '@platform/ui/toaster';
import { ArrowLeftRightIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { areVideoValuesEqual } from './videoComparators';
import { VideoComponentsSection } from './VideoComponentsSection';
import { VideoConceptsSection } from './VideoConceptsSection';
import { VideoConditioningClipField } from './VideoConditioningClipField';
import { VideoPromptFields } from './VideoFormFields';
import { VideoFrameImageField } from './VideoFrameImageField';
import { VideoLengthControls } from './VideoLengthControls';
import { VideoModelField } from './VideoModelField';
import { VideoReferenceListField } from './VideoReferenceListField';
import { VideoSourceClipField } from './VideoSourceClipField';
import { useVideoUi, useVideoUiActions } from './VideoUiContext';

/** Keep section props stable: project patches rerender this widget on every keystroke. */

const SWITCH_CHECKED_PROPS = { bg: 'accent.solid' };

const ASPECT_RATIO_COLLECTION = createListCollection({
  items: VIDEO_ASPECT_RATIO_IDS.map((id) => ({ label: id, value: id })),
});

const toTargetResolution = (value: string | undefined): VideoWidgetValues['targetResolution'] | null =>
  isVideoTargetResolution(value) ? value : null;

const DURATION_FORMATTER = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 2,
  minimumFractionDigits: 1,
});

/** Expose clearing for unsupported media that would otherwise block invocation from a hidden section. */
const StaleMediaStub = ({ label, onClear }: { label: string; onClear: () => void }) => {
  const { t } = useTranslation();

  return (
    <HStack bg="bg.subtle" gap="2" justify="space-between" p="2" rounded="md">
      <Text color="fg.muted" fontSize="xs" textWrap="pretty">
        {label}
      </Text>
      <Button flexShrink="0" size="sm" variant="outline" onClick={onClear}>
        {t('widgets.video.clearStaleMedia')}
      </Button>
    </HStack>
  );
};

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

    // Preserve topbar batchCount edits when seeding an otherwise uninitialized widget.
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
  const { patchValues, projectId, rawValues, readValues } = selection;
  // Reconcile only when inputs change; it is expensive and fresh values rerender every section.
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
  // What the run will actually use: a conditioning clip decides the length, and in the video role
  // the frame rate too. The stored values stay put underneath, so clearing the clip restores them.
  const timing = useMemo(() => getEffectiveVideoTiming(values.model ?? undefined, values), [values]);
  const autoDurationActive = useMemo(() => isAutoDurationActive(values), [values]);
  const autoDurationBounds = useMemo(() => getAutoDurationBounds(values.model ?? undefined, values), [values]);
  const autoDurationCeiling = useMemo(
    () =>
      autoDurationBounds
        ? {
            frames: autoDurationBounds.maxFrames,
            seconds: autoDurationBounds.maxFrames / autoDurationBounds.fps,
          }
        : null,
    [autoDurationBounds]
  );
  const autoDurationSupported = useMemo(() => isAutoDurationSupportedForMode(values), [values]);
  const extendNewFrames = autoDurationBounds
    ? autoDurationBounds.maxFrames - autoDurationBounds.contextFrames
    : (policy.ui.extendContext?.newFrames ?? 0);
  const durationSeconds = getVideoDurationSeconds(
    timing.numFrames,
    // In extend mode the extension inherits the SOURCE clip's frame rate.
    policy.fps.editable ? (values.sourceVideo?.fps ?? timing.fps) : policy.fps.defaultValue
  );

  const patch = useCallback((next: Partial<VideoWidgetValues>) => patchValues(next), [patchValues]);

  // Async writes must use the latest committed list for this project while the sync effect is live. Hidden
  // Activity panels stop syncing, and old callbacks still target their original project; drop writes in either
  // case to avoid restoring stale references. Two resolves in one microtask drain may still see the same
  // pre-commit list.
  const referencesRef = useRef({ live: true, projectId, references: values.references });

  useEffect(() => {
    referencesRef.current = { live: true, projectId, references: values.references };

    return () => {
      referencesRef.current = { ...referencesRef.current, live: false };
    };
  }, [projectId, values.references]);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

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

  // Stable per-field setters preserve child memo boundaries.
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
      audioCfgScale: (audioCfgScale: number) => patch({ audioCfgScale }),
      modalityScale: (modalityScale: number) => patch({ modalityScale }),
      stgScale: (stgScale: number) => patch({ stgScale }),
      fps: (fps: number) => patch({ fps }),
      autoDuration: ({ checked }: { checked: boolean }) => patch({ autoDuration: checked }),
      steps: (steps: number) => patch({ steps }),
      // Snapped on the way out: the VAE encodes 8k + 1 frames and the node snaps a ragged request
      // down silently, so an unsnapped value would leave the panel showing a number the run did
      // not use. The scrubber's own step keeps dragging on-grid; this covers typed input.
      ltx2ExtendContextFrames: (frames: number) =>
        patch({ ltx2ExtendContextFrames: Math.max(1 + LTX2_NUM_FRAMES_STEP, snapLtx2FramesDown(frames)) }),
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

  // First frame and initial video share one conditioning slot; last frame can accompany either. Reference
  // extension also keeps the linked tail reference synchronized.
  const referenceExtend = Boolean(policy.references?.extend);
  const maxVideoReferences = policy.references?.maxVideos ?? 3;
  const conditioningClip = values.conditioningClip;
  // A drop commits through the setter captured when it began, so what the frame displaces is read at commit time.
  const setFirstFrame = useMemo(
    () => createFrameImageSetter('firstFrameImage', readValues, patch),
    [patch, readValues]
  );
  const setLastFrame = useMemo(() => createFrameImageSetter('lastFrameImage', readValues, patch), [patch, readValues]);
  // Setting a frame is what clears a clip held for its picture, so the frame fields say so beforehand.
  const frameClearsClipText = isConditioningClipExcludingFrames(conditioningClip)
    ? t('widgets.video.frameClearsConditioningClip')
    : undefined;
  // A conditioning clip claims a whole modality, so it excludes the initial video and references, and in the picture
  // role the frames too -- and each of those clears it in turn. The role a dropped clip arrives in comes from the
  // gallery record: an uploaded soundtrack has no picture to condition on.
  const setConditioningClip = useCallback(
    (conditioningClip: VideoConditioningClip | null) => patch(getConditioningClipPatch(conditioningClip)),
    [patch]
  );
  // This setter tracks references separately from patch-only field setters. Rebudget the linked tail with frame
  // count so backend truncation cannot discard its seam end; derivation must tolerate intermediate input values.
  const setNumFrames = useCallback(
    (numFrames: number) =>
      patch(
        referenceExtend && referencesRef.current.live && referencesRef.current.projectId === projectId
          ? { numFrames, references: applyReferenceExtendNumFrames(referencesRef.current.references, numFrames) }
          : { numFrames }
      ),
    // Read committed references through the guarded ref to keep the Frames callback stable; rebudgeting is
    // idempotent.
    [patch, projectId, referenceExtend]
  );
  const setSourceVideo = useCallback(
    (sourceVideo: VideoSourceClip | null) => {
      if (referenceExtend && (!referencesRef.current.live || referencesRef.current.projectId !== projectId)) {
        return;
      }
      const placement = getInitialVideoPatch({
        maxVideos: maxVideoReferences,
        numFrames: values.numFrames,
        referenceExtend,
        references: referencesRef.current.references,
        sourceVideo,
      });

      // No tail-reference slot is available: refuse the whole drop so the clip cannot be set without its
      // continuity anchor.
      if (!placement) {
        toaster.create({
          description: t('widgets.video.referenceExtendCapFullDescription'),
          title: t('widgets.video.referenceExtendCapFull'),
          type: 'warning',
        });

        return;
      }
      patch(placement);
    },
    [maxVideoReferences, patch, projectId, referenceExtend, t, values.numFrames]
  );
  // Generation continues from the last reference; pin the continuity anchor after every list edit.
  const setReferences = useCallback(
    (update: (current: VideoReferenceItem[]) => VideoReferenceItem[]) => {
      if (!referencesRef.current.live || referencesRef.current.projectId !== projectId) {
        return;
      }
      const updated = update(referencesRef.current.references);

      // A declined/no-op updater must not reach the patch that clears frame slots and dirties the project.
      if (updated === referencesRef.current.references) {
        return;
      }
      patch(getReferencesPatch({ referenceExtend, references: updated }));
    },
    [patch, projectId, referenceExtend]
  );
  const clearReferences = useCallback(() => patch({ references: [] }), [patch]);
  const setLoras = useCallback(
    (update: (current: VideoWidgetValues['loras']) => VideoWidgetValues['loras']) => {
      let notice: Parameters<typeof toaster.create>[0] | undefined;
      patchValues((current) => {
        const loras = update(current.loras);

        if (loras === current.loras) {
          return {};
        }
        // While enabled, follow a replacement accelerator set or restore model sampling defaults if none remains.
        // Preserve the edit and notify; never enable acceleration from a list edit.
        if (!current.model) {
          return { loras };
        }

        const result = getAcceleratorLoraChangeResult(current, current.model, models, loras);

        if (result.outcome === 'switched') {
          notice = {
            description: t('widgets.video.acceleratorSwitchedDescription', {
              name: result.acceleratorLoras?.map((lora) => lora.name).join(', ') ?? '',
              steps: result.settings.steps,
            }),
            title: t('widgets.video.acceleratorSwitched', {
              label: getVideoModelPolicy(current.model, current).ui.accelerator?.label ?? '',
            }),
            type: 'info',
          };
        } else if (result.outcome === 'disabled') {
          notice = {
            description: t('widgets.video.acceleratorBrokenDescription'),
            title: t('widgets.video.acceleratorBroken'),
            type: 'info',
          };
        }

        return { ...result.settings };
      });

      if (notice) {
        toaster.create(notice);
      }
    },
    [models, patchValues, t]
  );
  const clearFirstFrame = useCallback(() => patch({ firstFrameImage: null }), [patch]);
  const clearLastFrame = useCallback(() => patch({ lastFrameImage: null }), [patch]);
  const clearSourceVideo = useCallback(() => patch({ sourceVideo: null }), [patch]);
  const clearConditioningClip = useCallback(() => patch({ conditioningClip: null }), [patch]);

  const targetResolutionCollection = useMemo(
    () => createListCollection({ items: policy.targetResolutions.map((option) => ({ ...option, value: option.id })) }),
    [policy.targetResolutions]
  );
  // A two-stage preset generates at half the canvas it names, which the size line below does not
  // say -- it reports the output size, which is the final one. Without this the whole signal that a
  // preset costs two passes is the three words in its own label.
  const twoStageHelpText = useMemo(() => {
    const option = policy.targetResolutions.find((entry) => entry.id === values.targetResolution);

    if (option?.stages !== 2 || !dimensions) {
      return undefined;
    }

    return t('widgets.video.twoStageHelp', {
      baseHeight: dimensions.height / 2,
      baseWidth: dimensions.width / 2,
      height: dimensions.height,
      width: dimensions.width,
    });
  }, [dimensions, policy.targetResolutions, t, values.targetResolution]);
  const aspectRatioValue = useMemo(() => [values.aspectRatioId], [values.aspectRatioId]);
  const targetResolutionValue = useMemo(() => [values.targetResolution], [values.targetResolution]);

  const framesSlider = useMemo(
    () =>
      policy.frames.kind === 'grid'
        ? {
            inputMax: policy.frames.max,
            max: policy.frames.max,
            min: policy.frames.min,
            step: policy.frames.step,
          }
        : {
            inputMax: policy.frames.choices[policy.frames.choices.length - 1] ?? 0,
            max: policy.frames.choices[policy.frames.choices.length - 1] ?? 0,
            min: policy.frames.choices[0] ?? 0,
            step:
              policy.frames.choices.length > 1 ? (policy.frames.choices[1] ?? 0) - (policy.frames.choices[0] ?? 0) : 1,
          },
    [policy.frames]
  );

  const hasAdvancedGuidance = policy.ui.audioCfgVisible || policy.ui.stgVisible || policy.ui.modalityVisible;
  const mode = resolveVideoMode(values);
  const supportsFirstFrame = policy.modes.includes('first-frame') || policy.modes.includes('first-last');
  const supportsLastFrame = policy.modes.includes('first-last') || policy.modes.includes('last-frame');
  const promptEnhancer = policy.prompt.enhancer;
  const firstFrameImage = values.firstFrameImage;
  // Memoized for VideoPromptFields' equality check; both inputs keep their identity between patches.
  const expandPromptSuggestion = useMemo(
    () => getVideoExpandPromptSuggestion(promptEnhancer, firstFrameImage),
    [promptEnhancer, firstFrameImage]
  );
  const supportsExtend = policy.modes.includes('extend');
  const supportsReferences = policy.modes.includes('reference');
  const supportsConditioningClip = policy.modes.includes('audio-to-video') || policy.modes.includes('video-to-audio');
  const supportsInitialVideo = supportsExtend || referenceExtend;
  // Use the setter's capacity predicate: recalled unflagged references can be adopted without consuming a slot,
  // and must not disable clip trimming.
  const initialVideoCapBlocked =
    referenceExtend &&
    !canPlaceReferenceExtendAnchor(values.references, values.sourceVideo?.video_name, maxVideoReferences);
  // The aspect-ratio control is locked by media that pins the frame. A clip in the `audio` role
  // does not: its picture is what gets generated, so the ratio is still the user's to choose.
  const hasConditioningMedia = Boolean(
    values.firstFrameImage || values.lastFrameImage || values.sourceVideo || values.conditioningClip?.role === 'video'
  );
  const otherMediaSet = Boolean(values.sourceVideo || values.references.length > 0);
  // Frames block only the clip's picture role; its soundtrack is what they can be combined with.
  const framesSet = Boolean(values.firstFrameImage || values.lastFrameImage);
  const conditioningDerivedText = values.conditioningClip
    ? t(
        values.conditioningClip.role === 'audio'
          ? 'widgets.video.conditioningDerivedAudio'
          : 'widgets.video.conditioningDerivedVideo',
        { fps: timing.fps, frames: timing.numFrames }
      )
    : undefined;
  const derivedSourceText = dimensions ? t(`widgets.video.dimensionSource.${dimensions.source}`) : undefined;
  // Media determines output proportions. Show its ratio source while disabled, preserving the saved preset for
  // when media is removed.
  const dimensionSource = dimensions?.source;
  const derivedSourceValueText = useMemo(
    () =>
      dimensionSource && dimensionSource !== 'aspect-ratio' ? (
        <Text as="span" fontSize="md" truncate>
          {t(`widgets.video.dimensionSourceValue.${dimensionSource}`)}
        </Text>
      ) : undefined,
    [dimensionSource, t]
  );
  const derivedSizeText = dimensions
    ? `${t('widgets.video.derivedSize', { height: dimensions.height, width: dimensions.width })}${
        derivedSourceText ? ` — ${derivedSourceText}` : ''
      }`
    : t('widgets.video.derivedSizeUnavailable');
  const fpsLockedForExtend = policy.ui.fpsVisible && mode === 'extend';
  const fpsLocked = fpsLockedForExtend || timing.fpsFromClip;
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

      <Stack gap="1" px="2" py="1">
        <VideoModelField
          models={models}
          modelsLoaded={modelsStatus === 'loaded'}
          projectId={projectId}
          values={values}
        />
      </Stack>

      <VideoPromptFields
        expandPromptSuggestion={expandPromptSuggestion}
        loras={values.loras}
        model={values.model}
        negativeHelpText={policy.prompt.negativeHelpTextKey ? t(policy.prompt.negativeHelpTextKey) : undefined}
        negativePrompt={values.negativePrompt}
        negativePromptEnabled={values.negativePromptEnabled}
        negativePromptHeightPx={values.negativePromptHeightPx}
        negativeVisible={policy.prompt.negativeVisible}
        positivePrompt={values.positivePrompt}
        positivePromptHeightPx={values.positivePromptHeightPx}
        projectId={projectId}
        showSyntaxHighlighting={selection.showPromptSyntaxHighlighting}
        onPatchValues={patch}
      />

      {supportsFirstFrame || supportsLastFrame ? (
        <GenerationSettingsSection label={t('widgets.video.initialFrames')} sectionId="video-frames" defaultOpen>
          <Stack gap="3" p="2">
            {supportsFirstFrame ? (
              <Field
                helpText={
                  values.sourceVideo
                    ? undefined
                    : [t('widgets.video.firstFrameHelp'), frameClearsClipText].filter(Boolean).join(' ')
                }
                label={t('widgets.video.firstFrame')}
              >
                <VideoFrameImageField
                  disabled={Boolean(values.sourceVideo)}
                  disabledReason={values.sourceVideo ? t('widgets.video.firstFrameBlocked') : undefined}
                  dropId="video-first-frame"
                  dropLabel={t('widgets.video.dropFirstFrame')}
                  image={values.firstFrameImage}
                  onChange={setFirstFrame}
                />
              </Field>
            ) : null}
            {supportsLastFrame ? (
              <Field
                helpText={[
                  values.sourceVideo ? t('widgets.video.lastFrameExtendHelp') : t('widgets.video.lastFrameHelp'),
                  frameClearsClipText,
                ]
                  .filter(Boolean)
                  .join(' ')}
                label={t('widgets.video.lastFrame')}
              >
                <VideoFrameImageField
                  dropId="video-last-frame"
                  dropLabel={t('widgets.video.dropLastFrame')}
                  image={values.lastFrameImage}
                  onChange={setLastFrame}
                />
              </Field>
            ) : null}
          </Stack>
        </GenerationSettingsSection>
      ) : null}

      {!supportsFirstFrame && values.firstFrameImage ? (
        <StaleMediaStub label={t('widgets.video.staleFirstFrame')} onClear={clearFirstFrame} />
      ) : null}
      {!supportsLastFrame && values.lastFrameImage ? (
        <StaleMediaStub label={t('widgets.video.staleLastFrame')} onClear={clearLastFrame} />
      ) : null}
      {!supportsInitialVideo && values.sourceVideo ? (
        <StaleMediaStub label={t('widgets.video.staleSourceVideo')} onClear={clearSourceVideo} />
      ) : null}
      {!supportsReferences && values.references.length > 0 ? (
        <StaleMediaStub label={t('widgets.video.staleReferences')} onClear={clearReferences} />
      ) : null}
      {!supportsConditioningClip && values.conditioningClip ? (
        <StaleMediaStub label={t('widgets.video.staleConditioningClip')} onClear={clearConditioningClip} />
      ) : null}

      {supportsReferences ? (
        <GenerationSettingsSection label={t('widgets.video.references')} sectionId="video-references" defaultOpen>
          <Stack gap="3" p="2">
            <VideoReferenceListField
              maxImages={policy.references?.maxImages ?? 9}
              maxVideos={policy.references?.maxVideos ?? 3}
              references={values.references}
              targetArea={dimensions ? dimensions.width * dimensions.height : null}
              onChange={setReferences}
            />
          </Stack>
        </GenerationSettingsSection>
      ) : null}

      {supportsInitialVideo ? (
        <GenerationSettingsSection label={t('widgets.video.initialVideo')} sectionId="video-source" defaultOpen>
          <Stack gap="3" p="2">
            {referenceExtend ? (
              <Text color="fg.muted" fontSize="xs" textWrap="pretty">
                {t('widgets.video.referenceExtendHelp')}
              </Text>
            ) : null}
            <VideoSourceClipField
              disabled={Boolean(values.firstFrameImage) || initialVideoCapBlocked}
              disabledReason={
                values.firstFrameImage
                  ? t('widgets.video.initialVideoBlocked')
                  : initialVideoCapBlocked
                    ? t('widgets.video.initialVideoCapBlocked')
                    : undefined
              }
              sourceVideo={values.sourceVideo}
              onChange={setSourceVideo}
            />
            {policy.ui.extendContext ? (
              <ScrubberField
                defaultValue={LTX2_EXTEND_CONTEXT_FRAMES}
                // The trade this control makes, which Frames alone does not show: the join consumes
                // the context from both halves, so every frame held is a frame of new video given up.
                // Under auto duration Frames is only a ceiling, so the new material is too.
                helpText={t(
                  autoDurationBounds ? 'widgets.video.extendContextHelpAuto' : 'widgets.video.extendContextHelp',
                  {
                    frames: extendNewFrames,
                    // The continuation plays at the source's rate, not the panel's.
                    seconds: (extendNewFrames / Math.max(1, values.sourceVideo?.fps ?? values.fps)).toFixed(1),
                  }
                )}
                inputMax={policy.ui.extendContext.max}
                label={t('widgets.video.extendContext')}
                max={policy.ui.extendContext.max}
                min={policy.ui.extendContext.min}
                step={policy.ui.extendContext.step}
                value={policy.ui.extendContext.value}
                onChange={set.ltx2ExtendContextFrames}
              />
            ) : null}
          </Stack>
        </GenerationSettingsSection>
      ) : null}

      {supportsConditioningClip ? (
        <GenerationSettingsSection
          label={t('widgets.video.conditioningClip')}
          sectionId="video-conditioning-clip"
          defaultOpen={Boolean(values.conditioningClip)}
        >
          <Stack gap="3" p="2">
            <VideoConditioningClipField
              conditioningClip={values.conditioningClip}
              derivedText={conditioningDerivedText}
              disabled={otherMediaSet}
              disabledReason={otherMediaSet ? t('widgets.video.conditioningClipBlocked') : undefined}
              pictureRoleDisabledReason={framesSet ? t('widgets.video.conditioningRoleVideoBlocked') : undefined}
              onChange={setConditioningClip}
            />
          </Stack>
        </GenerationSettingsSection>
      ) : null}

      <GenerationSettingsSection label={t('widgets.video.dimensions')} sectionId="video-dimensions" defaultOpen>
        <Stack gap="3" p="2">
          <Field helpText={derivedSizeText} label={t('widgets.video.aspectRatio')}>
            <HStack gap="1">
              <Select
                collection={ASPECT_RATIO_COLLECTION}
                disabled={hasConditioningMedia}
                flex="1"
                value={aspectRatioValue}
                valueText={derivedSourceValueText}
                onValueChange={set.aspectRatio}
              />
              <IconButton
                aria-label={t('widgets.video.swapAspectRatio')}
                disabled={hasConditioningMedia}
                variant="ghost"
                onClick={swapAspectRatio}
              >
                <ArrowLeftRightIcon />
              </IconButton>
            </HStack>
          </Field>
          <Field helpText={twoStageHelpText} label={t('widgets.video.targetResolution')}>
            <Select
              collection={targetResolutionCollection}
              value={targetResolutionValue}
              onValueChange={set.targetResolution}
            />
          </Field>
          <VideoLengthControls
            autoDuration={values.autoDuration}
            autoDurationActive={autoDurationActive}
            autoDurationCeiling={autoDurationCeiling}
            autoDurationSupported={autoDurationSupported}
            durationText={durationText}
            framesSlider={framesSlider}
            hasDurationHead={Boolean(values.ltx2DurationHeadModel)}
            numFrames={timing.numFrames}
            numFramesFromClip={timing.numFramesFromClip}
            onAutoDurationChange={set.autoDuration}
            onNumFramesChange={setNumFrames}
          />
          {policy.ui.fpsVisible ? (
            <ScrubberField
              disabled={fpsLocked}
              helpText={
                timing.fpsFromClip
                  ? t('widgets.video.fpsFromClip')
                  : fpsLockedForExtend
                    ? t('widgets.video.fpsExtendLocked')
                    : undefined
              }
              inputMax={policy.fps.max}
              label={t('widgets.video.fps')}
              max={60}
              min={policy.fps.min}
              step={1}
              value={timing.fps}
              onChange={set.fps}
            />
          ) : (
            <Text color="fg.muted" fontSize="xs">
              {t('widgets.video.fixedFps', { fps: policy.fps.defaultValue })}
            </Text>
          )}
        </Stack>
      </GenerationSettingsSection>

      {/* Sampling and variation — how the model renders, not which model it is. */}
      <GenerationSettingsSection label={t('widgets.video.render')} sectionId="video-render" defaultOpen>
        <Stack gap="3" p="2">
          {policy.ui.accelerator && values.model ? (
            <Field
              helpText={t('widgets.video.acceleratorHelp', {
                label: policy.ui.accelerator.label,
                steps: policy.ui.acceleratorSteps ?? policy.ui.accelerator.steps,
              })}
              label={t('widgets.video.accelerator', { label: policy.ui.accelerator.label })}
            >
              <Switch.Root checked={values.acceleratorEnabled} onCheckedChange={toggleAccelerator}>
                <Switch.HiddenInput />
                <Switch.Control _checked={SWITCH_CHECKED_PROPS}>
                  <Switch.Thumb />
                </Switch.Control>
              </Switch.Root>
            </Field>
          ) : null}
          {policy.ui.stepsEditable ? (
            <ScrubberField
              defaultValue={policy.defaults.steps}
              hint="steps"
              inputMax={500}
              label={t('widgets.video.steps')}
              max={100}
              min={policy.minSteps}
              step={1}
              value={values.steps}
              onChange={set.steps}
            />
          ) : (
            <Text color="fg.muted" fontSize="xs">
              {t('widgets.video.stepsFixed', { steps: policy.defaults.steps })}
            </Text>
          )}
          {policy.ui.cfgVisible ? (
            <ScrubberField
              hint="cfgScale"
              inputMax={100}
              label={t('widgets.video.cfg')}
              max={15}
              min={1}
              step={0.1}
              value={values.cfgScale}
              onChange={set.cfgScale}
            />
          ) : null}
          {policy.ui.cfgLowNoiseVisible ? (
            <ScrubberField
              helpText={t('widgets.video.cfgLowNoiseHelp')}
              inputMax={100}
              label={t('widgets.video.cfgLowNoise')}
              max={15}
              min={1}
              step={0.1}
              value={values.cfgScaleLowNoise ?? values.cfgScale}
              onChange={set.cfgScaleLowNoise}
            />
          ) : null}
          <SeedField
            batchCount={values.batchCount}
            label={t('widgets.video.seed')}
            seed={values.seed}
            seedMode={values.seedMode}
            onCommit={patch}
          />
        </Stack>
      </GenerationSettingsSection>

      {hasAdvancedGuidance ? (
        <GenerationSettingsSection label={t('widgets.video.advancedGuidance')} sectionId="video-guidance">
          <Stack gap="3" p="2">
            {policy.ui.audioCfgVisible ? (
              <ScrubberField
                defaultValue={policy.defaults.audioCfgScale ?? undefined}
                helpText={t('widgets.video.audioCfgHelp')}
                inputMax={100}
                label={t('widgets.video.audioCfg')}
                max={15}
                min={1}
                step={0.1}
                value={values.audioCfgScale ?? policy.defaults.audioCfgScale ?? 1}
                onChange={set.audioCfgScale}
              />
            ) : null}
            {policy.ui.stgVisible ? (
              <ScrubberField
                defaultValue={policy.defaults.stgScale ?? undefined}
                helpText={t('widgets.video.stgHelp')}
                inputMax={10}
                label={t('widgets.video.stg')}
                max={3}
                min={0}
                step={0.1}
                value={values.stgScale ?? policy.defaults.stgScale ?? 0}
                onChange={set.stgScale}
              />
            ) : null}
            {policy.ui.modalityVisible ? (
              <ScrubberField
                defaultValue={policy.defaults.modalityScale ?? undefined}
                helpText={t('widgets.video.modalityHelp')}
                inputMax={10}
                label={t('widgets.video.modality')}
                max={5}
                min={1}
                step={0.1}
                value={values.modalityScale ?? policy.defaults.modalityScale ?? 1}
                onChange={set.modalityScale}
              />
            ) : null}
          </Stack>
        </GenerationSettingsSection>
      ) : null}

      <VideoConceptsSection projectId={projectId} loras={values.loras} model={values.model} onChangeLoras={setLoras} />
      <VideoComponentsSection values={values} onPatch={patch} />
    </Stack>
  );
};
