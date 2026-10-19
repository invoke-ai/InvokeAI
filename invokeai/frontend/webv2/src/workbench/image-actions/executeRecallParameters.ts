import type { ComponentModelConfig, VaeModelConfig } from '@features/generation/contracts';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { getEffectiveReferenceImage, isSupportedGenerateModel, isVaeModelConfig } from '@features/generation/settings';
import { assertAccountScopeCurrent, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';

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
  getGenerateValues,
  models,
  owner,
  parameters,
  projectId,
  t,
}: {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
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
        title: 'Cannot apply recalled parameters',
      });
      return false;
    }

    if (!isRecallParametersAvailable(parameters)) {
      commands.notifications.add({
        kind: 'info',
        message: t('widgets.generate.capabilitiesUnavailableForRecall'),
        title: 'Cannot apply recalled parameters',
      });
      return false;
    }

    const result = buildRecallParametersSettings({ currentValues, models, parameters, supportedModels, vaeModels });

    if (result.fields.includes('referenceImages')) {
      const existingIds = new Set(currentValues.referenceImages.map((referenceImage) => referenceImage.id));
      await verifyRecalledReferenceImages(result, existingIds, owner.signal);
      assertAccountScopeCurrent(owner);

      if (getGenerateValues() === null) {
        return false;
      }
    }

    const skipMessage = result.skipped.length > 0 ? getRecallParametersSkipMessage(result.skipped) : null;

    if (result.fields.length === 0) {
      commands.notifications.add({
        kind: 'info',
        message: skipMessage ?? 'The request did not include any parameters the Generate panel uses.',
        title: 'No recalled parameters applied',
      });
      return false;
    }

    commands.generation.setSettings(result.values, projectId);
    commands.notifications.add({
      kind: 'success',
      message: getRecallParametersMessage(result.fields),
      title: 'Recalled parameters',
    });
    if (skipMessage) {
      commands.notifications.add({
        kind: 'info',
        message: skipMessage,
        title: 'Some recalled parameters were not applied',
      });
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
