import type { ComponentModelConfig, VaeModelConfig } from '@features/generation/contracts';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { getEffectiveReferenceImage, isSupportedGenerateModel, isVaeModelConfig } from '@features/generation/settings';
import { assertAccountScopeCurrent, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';
import { DEFAULT_CANVAS_DENOISING_STRENGTH } from '@workbench/widgets/canvas/invoke/canvasStrength';

import type { CanvasRecallTarget } from './canvasRecall';

import {
  filterAvailableReferenceImages,
  getCurrentGenerateValues,
  getMissingGenerateValuesMessage,
} from './executeImageRecall';
import {
  buildRecallParametersSettings,
  getRecallParametersMessage,
  getRecallParametersSkipMessage,
  isRecallParametersAvailable,
  type RecallParametersResult,
} from './recallParameters';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

/**
 * Drops recalled reference images whose files are not in the gallery, reporting
 * each one. Entries the project already had are trusted as-is, so an `append`
 * never removes anything. When nothing recalled survives, the project keeps
 * the list it had instead of committing an empty replacement.
 */
const verifyRecalledReferenceImages = async (
  result: RecallParametersResult,
  existingIds: ReadonlySet<string>,
  signal: AbortSignal
): Promise<void> => {
  const recalled = result.values.referenceImages.filter((referenceImage) => !existingIds.has(referenceImage.id));

  if (recalled.length === 0) {
    return;
  }

  const availableIds = new Set(
    (await filterAvailableReferenceImages(recalled, signal)).map((referenceImage) => referenceImage.id)
  );

  for (const referenceImage of recalled) {
    if (!availableIds.has(referenceImage.id)) {
      const asset = referenceImage.config.image;
      result.skipped.push({
        key: 'reference_images',
        reason: 'unresolved',
        ...(asset ? { detail: getEffectiveReferenceImage(asset).image_name } : {}),
      });
    }
  }

  if (availableIds.size === 0) {
    result.values.referenceImages = result.retainedReferenceImages;
    result.fields = result.fields.filter((field) => field !== 'referenceImages');
    return;
  }

  result.values.referenceImages = result.values.referenceImages.filter(
    (referenceImage) => existingIds.has(referenceImage.id) || availableIds.has(referenceImage.id)
  );
};

/** Apply one recall payload; stop quietly if the project disappears. Return true when at least one field commits. */
export const executeRecallParameters = async ({
  commands,
  canvasTarget,
  getGenerateValues,
  models,
  owner,
  parameters,
  projectId,
  t,
}: {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  canvasTarget?: CanvasRecallTarget;
  getGenerateValues: () => Record<string, unknown> | null;
  models: readonly ComponentModelConfig[];
  owner: AccountScope;
  parameters: Record<string, unknown>;
  projectId: string;
  t: TFunction;
}): Promise<boolean> => {
  if (!isAccountScopeCurrent(owner)) {
    return false;
  }

  const originalProject = canvasTarget?.getProject();
  if (canvasTarget && !originalProject) {
    return false;
  }

  const supportedModels = models.filter(isSupportedGenerateModel);
  const vaeModels = models.filter(isVaeModelConfig).map((model) => model as VaeModelConfig);

  try {
    const generateValues = getGenerateValues();

    if (generateValues === null) {
      return false;
    }

    const currentValues = getCurrentGenerateValues({ generateValues, supportedModels });

    if (!currentValues) {
      commands.notifications.add({
        kind: 'info',
        message: getMissingGenerateValuesMessage(t),
        title: t('widgets.gallery.itemActions.recall.parameters.cannotApply'),
      });
      return false;
    }

    if (!isRecallParametersAvailable(parameters)) {
      commands.notifications.add({
        kind: 'info',
        message: t('widgets.generate.capabilitiesUnavailableForRecall'),
        title: t('widgets.gallery.itemActions.recall.parameters.cannotApply'),
      });
      return false;
    }

    let result = buildRecallParametersSettings({ currentValues, models, parameters, supportedModels, vaeModels });

    let availableReferenceNames: Set<string> | null = null;
    if (result.fields.includes('referenceImages')) {
      const existingIds = new Set(currentValues.referenceImages.map((referenceImage) => referenceImage.id));
      await verifyRecalledReferenceImages(result, existingIds, owner.signal);
      availableReferenceNames = new Set(
        result.values.referenceImages
          .filter((ref) => !existingIds.has(ref.id))
          .flatMap((ref) => (ref.config.image ? [getEffectiveReferenceImage(ref.config.image).image_name] : []))
      );
      assertAccountScopeCurrent(owner);

      if (getGenerateValues() === null) {
        return false;
      }
    }

    const canvasRecall =
      canvasTarget && originalProject && ('control_layers' in parameters || 'denoise_strength' in parameters)
        ? await import('./canvasRecall')
        : null;
    const canvasPlan = canvasRecall
      ? await canvasRecall.prepareCanvasRecall({
          metadata: {
            ...parameters,
            ...(parameters.denoise_strength === null ? { denoise_strength: DEFAULT_CANVAS_DENOISING_STRENGTH } : {}),
            ...(parameters.control_layers === null ? { control_layers: [] } : {}),
          },
          model: result.values.model,
          models,
          owner,
          isCurrent: () => canvasTarget?.getProject()?.canvas.document === originalProject?.canvas.document,
        })
      : null;
    if (canvasPlan && canvasRecall && canvasTarget) {
      await canvasRecall.settleCanvasRecall(canvasPlan, canvasTarget, owner);
    }
    assertAccountScopeCurrent(owner);
    const latestRaw = getGenerateValues();
    if (latestRaw === null || (canvasTarget && !canvasTarget.getProject())) {
      return false;
    }
    const latest = getCurrentGenerateValues({ generateValues: latestRaw, supportedModels });
    if (!latest) {
      return false;
    }
    // Rebuild after every awaited lookup so newer prompt/model edits are not overwritten by an old snapshot.
    const previous = result;
    result = buildRecallParametersSettings({ currentValues: latest, models, parameters, supportedModels, vaeModels });
    if (availableReferenceNames !== null && result.fields.includes('referenceImages')) {
      const existingIds = new Set(latest.referenceImages.map((ref) => ref.id));
      const kept = result.values.referenceImages.filter(
        (ref) =>
          existingIds.has(ref.id) ||
          (ref.config.image && availableReferenceNames.has(getEffectiveReferenceImage(ref.config.image).image_name))
      );
      const anyRecalled = kept.some((ref) => !existingIds.has(ref.id));
      if (
        !anyRecalled &&
        ((Array.isArray(parameters.ip_adapters) && parameters.ip_adapters.length > 0) ||
          (Array.isArray(parameters.reference_images) && parameters.reference_images.length > 0))
      ) {
        result.values.referenceImages = result.retainedReferenceImages;
        result.fields = result.fields.filter((field) => field !== 'referenceImages');
      } else {
        result.values.referenceImages = kept;
      }
      result.skipped.push(
        ...previous.skipped.filter((skip) => skip.key === 'reference_images' && skip.reason === 'unresolved')
      );
    }
    let canvasApplied = false;
    if (canvasPlan && canvasRecall && canvasTarget && originalProject) {
      if (previous.values.model.key !== result.values.model.key) {
        canvasPlan.problems.push('model-changed');
      } else {
        canvasApplied = canvasRecall.commitCanvasRecall({
          plan: canvasPlan,
          target: canvasTarget,
          expectedDocument: originalProject.canvas.document,
          append: parameters.append === true,
          restorePointName: t('widgets.gallery.itemActions.recall.beforeCanvasRecall'),
        });
      }
      result.skipped.push(
        ...canvasPlan.problems.map((detail) => ({
          key: 'control_layers',
          reason: 'unresolved' as const,
          detail: canvasRecall.getCanvasRecallProblemsMessage([detail], t),
        }))
      );
    }
    if (canvasApplied) {
      result.fields.push('canvas');
    }
    if (!canvasTarget) {
      for (const key of ['control_layers', 'denoise_strength']) {
        if (key in parameters) {
          result.skipped.push({ key, reason: 'unsupported' });
        }
      }
    }
    const skipMessage = result.skipped.length > 0 ? getRecallParametersSkipMessage(result.skipped, t) : null;

    if (result.fields.length === 0) {
      commands.notifications.add({
        kind: 'info',
        message: skipMessage ?? t('widgets.gallery.itemActions.recall.parameters.nothingAppliedDescription'),
        title: t('widgets.gallery.itemActions.recall.parameters.nothingApplied'),
      });
      return false;
    }

    if (result.fields.some((field) => field !== 'canvas')) {
      commands.generation.setSettings(result.values, projectId);
    }
    commands.notifications.add({
      kind: 'success',
      message: getRecallParametersMessage(result.fields, t),
      title: t('widgets.gallery.itemActions.recall.parameters.applied'),
    });
    if (skipMessage) {
      commands.notifications.add({
        kind: 'info',
        message: skipMessage,
        title: t('widgets.gallery.itemActions.recall.parameters.partiallyApplied'),
      });
    }
    if (canvasApplied && canvasPlan && canvasRecall && canvasTarget) {
      const durable = await canvasRecall.finalizeCanvasRecallAssets(canvasPlan, canvasTarget, owner);
      if (!durable && isAccountScopeCurrent(owner) && canvasTarget.getProject()) {
        commands.notifications.add({
          kind: 'info',
          title: t('widgets.gallery.itemActions.recall.parameters.partiallyApplied'),
          message: t('widgets.gallery.itemActions.recall.canvasImagesTemporary'),
        });
      }
    }
    return true;
  } catch (error: unknown) {
    if (!isAccountScopeCurrent(owner)) {
      return false;
    }

    commands.notifications.reportError({
      area: 'recall-parameters',
      message: toErrorMessage(error),
      namespace: 'generation',
      projectId,
    });
    return false;
  }
};
