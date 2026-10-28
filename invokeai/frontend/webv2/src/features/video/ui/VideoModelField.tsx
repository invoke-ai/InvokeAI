/* oxlint-disable react-perf/jsx-no-new-function-as-prop -- the React Compiler memoizes these handlers, as in GenerateModelCard. */
import type { MainModelConfig } from '@features/generation/contracts';
import type { ModelConfig, ModelTaxonomyType } from '@features/models';
import type { VideoWidgetValues } from '@features/video/core/types';
import type {
  VideoModelSelectionResult,
  VideoModelSwitchAdjustment,
  VideoModelSwitchLossKey,
} from '@features/video/core/videoPolicies';

import { Box, Text } from '@chakra-ui/react';
import { isMainModelConfig } from '@features/generation/settings';
import { ModelSelect } from '@features/models/react';
import { getVideoModelSelectionResult, isVideoModelSelectable } from '@features/video/core/videoPolicies';
import { syncVideoWidgetValuesWithModels } from '@features/video/core/widgetValues';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { Field } from '@platform/ui';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { toaster } from '@platform/ui/toaster';
import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useVideoUiActions } from './VideoUiContext';

const MAIN_MODEL_TYPES: readonly ModelTaxonomyType[] = ['main'];

// `labelKey` is spelled out per entry so the translation-key scan checks each one.
const LOSS_LABELS: Record<VideoModelSwitchLossKey, { labelKey: string }> = {
  componentSourceModel: { labelKey: 'widgets.video.switchModelLosses.componentSourceModel' },
  conditioningClip: { labelKey: 'widgets.video.switchModelLosses.conditioningClip' },
  firstFrame: { labelKey: 'widgets.video.switchModelLosses.firstFrame' },
  h3HybridBaseModel: { labelKey: 'widgets.video.switchModelLosses.h3HybridBaseModel' },
  h3TextEncoderModel: { labelKey: 'widgets.video.switchModelLosses.h3TextEncoderModel' },
  h3TransformerModel: { labelKey: 'widgets.video.switchModelLosses.h3TransformerModel' },
  initialVideo: { labelKey: 'widgets.video.switchModelLosses.initialVideo' },
  lastFrame: { labelKey: 'widgets.video.switchModelLosses.lastFrame' },
  loras: { labelKey: 'widgets.video.switchModelLosses.loras' },
  ltx2DurationHeadModel: { labelKey: 'widgets.video.switchModelLosses.ltx2DurationHeadModel' },
  ltx2TextEncoderModel: { labelKey: 'widgets.video.switchModelLosses.ltx2TextEncoderModel' },
  references: { labelKey: 'widgets.video.switchModelLosses.references' },
  trimmedInitialVideo: { labelKey: 'widgets.video.switchModelLosses.trimmedInitialVideo' },
  vae: { labelKey: 'widgets.video.switchModelLosses.vae' },
  wanLowNoiseModel: { labelKey: 'widgets.video.switchModelLosses.wanLowNoiseModel' },
  wanT5EncoderModel: { labelKey: 'widgets.video.switchModelLosses.wanT5EncoderModel' },
};

// Adjusted values are named as their controls are labelled, so they can be found on screen.
const ADJUSTMENT_LABELS: Record<VideoModelSwitchAdjustment, { labelKey: string }> = {
  acceleration: { labelKey: 'widgets.video.acceleration' },
  advancedGuidance: { labelKey: 'widgets.video.advancedGuidance' },
  autoDuration: { labelKey: 'widgets.video.autoDuration' },
  cfgLowNoise: { labelKey: 'widgets.video.cfgLowNoise' },
  fps: { labelKey: 'widgets.video.fps' },
  frames: { labelKey: 'widgets.video.frames' },
  steps: { labelKey: 'widgets.video.steps' },
  targetResolution: { labelKey: 'widgets.video.targetResolution' },
};

const findSelectableModel = (models: readonly ModelConfig[], key: string) => {
  const model = models.find((candidate) => candidate.key === key) ?? null;

  return isMainModelConfig(model) && isVideoModelSelectable(model) ? model : null;
};

/**
 * The video model picker. A switch that would discard user-supplied setup waits for confirmation; any other switch
 * applies at once and reports the values it adjusted.
 */
export const VideoModelField = ({
  models,
  modelsLoaded,
  projectId,
  values,
}: {
  models: readonly ModelConfig[];
  modelsLoaded: boolean;
  projectId: string;
  values: VideoWidgetValues;
}) => {
  const { i18n, t } = useTranslation();
  const { patchValues } = useVideoUiActions();
  // The picker keeps its default id, which keys its remembered view and base filter; focus returns through this.
  const pickerRef = useRef<HTMLDivElement>(null);
  // Retain only the target and its project; the dialog previews, and confirmation applies, against live settings.
  const [pendingSwitch, setPendingSwitch] = useState<{ modelKey: string; projectId: string } | null>(null);
  const pendingModel =
    pendingSwitch?.projectId === projectId ? findSelectableModel(models, pendingSwitch.modelKey) : null;

  // A target uninstalled while the dialog is open, or a project switch under it, closes it without applying.
  if (pendingSwitch && !pendingModel) {
    setPendingSwitch(null);
  }

  // One transition for the preview, the decision, and the confirmed write, so they cannot disagree. The view shows
  // panel values reconciled with the catalog; stored values are judged the same way.
  const getTransition = (current: VideoWidgetValues, model: MainModelConfig) =>
    getVideoModelSelectionResult({
      currentSettings: modelsLoaded ? syncVideoWidgetValuesWithModels(current, models) : current,
      model,
      models,
    });
  const preview = pendingModel ? getTransition(values, pendingModel) : null;

  const formatList = (labels: string[]) =>
    new Intl.ListFormat(i18n.resolvedLanguage, { style: 'long', type: 'conjunction' }).format(labels);
  const formatAdjustments = (adjustments: readonly VideoModelSwitchAdjustment[]) =>
    formatList(adjustments.map((adjustment) => t(ADJUSTMENT_LABELS[adjustment].labelKey)));

  const selectModel = (candidate: ModelConfig | null) => {
    if (!isMainModelConfig(candidate) || !isVideoModelSelectable(candidate)) {
      return;
    }

    // Commit pending concept and prompt drafts first, so the switch is judged against, and keeps, what they hold.
    flushWorkbenchDrafts();

    let result = undefined as VideoModelSelectionResult | undefined;

    // Decide and apply in one functional patch, against the latest stored values rather than this render's.
    patchValues((current) => {
      result = getTransition(current, candidate);

      return result.losses.length > 0 ? {} : { ...result.settings, model: candidate };
    });

    if (!result) {
      return;
    }

    if (result.losses.length > 0) {
      setPendingSwitch({ modelKey: candidate.key, projectId });
    } else if (result.adjustments.length > 0) {
      toaster.create({
        description: t('widgets.video.settingsAdjustedDescription', { labels: formatAdjustments(result.adjustments) }),
        title: t('widgets.video.settingsAdjusted'),
        type: 'info',
      });
    }
  };

  const confirmSwitch = () => {
    if (!pendingModel) {
      return;
    }

    flushWorkbenchDrafts();
    // Close before writing, so the dialog animates out showing the list that was confirmed, not the switched panel.
    setPendingSwitch(null);
    // Recompute against the settings as they stand now: they may have changed while the dialog was open.
    patchValues((current) => ({ ...getTransition(current, pendingModel).settings, model: pendingModel }));
  };

  return (
    <>
      <Field
        error={values.model ? undefined : t('widgets.video.modelRequired')}
        hint="model"
        label={t('widgets.video.mainModel')}
      >
        <Box ref={pickerRef} minW="0" w="full">
          <ModelSelect
            filter={isVideoModelSelectable}
            invalid={!values.model}
            modelTypes={MAIN_MODEL_TYPES}
            placeholder={t('widgets.video.selectModel')}
            value={values.model?.key ?? null}
            onChange={selectModel}
          />
        </Box>
      </Field>

      <ConfirmDialog
        body={
          preview ? (
            <>
              {preview.losses.length > 0 ? (
                <Text fontSize="lg">
                  {t('widgets.video.switchModelBody', {
                    labels: formatList(preview.losses.map(({ count, key }) => t(LOSS_LABELS[key].labelKey, { count }))),
                    name: pendingModel?.name ?? '',
                  })}
                </Text>
              ) : null}
              {preview.adjustments.length > 0 ? (
                <Text color="fg.muted" fontSize="md">
                  {t('widgets.video.switchModelAdjusts', { labels: formatAdjustments(preview.adjustments) })}
                </Text>
              ) : null}
            </>
          ) : null
        }
        confirmLabel={t('widgets.video.switchModelConfirm')}
        finalFocusEl={() => pickerRef.current?.querySelector<HTMLElement>('[aria-haspopup="listbox"]') ?? null}
        isOpen={pendingModel !== null}
        title={t('widgets.video.switchModelTitle')}
        onClose={() => setPendingSwitch(null)}
        onConfirm={confirmSwitch}
      />
    </>
  );
};
