import type { GalleryImage } from '@features/gallery';
import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateWidgetValues,
  VaeModelConfig,
} from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { galleryImages } from '@features/gallery';
import {
  getDefaultGenerateSettings,
  getEffectiveReferenceImage,
  hasArchitectureCapabilities,
  isSupportedGenerateModel,
  isVaeModelConfig,
  normalizeGenerateWidgetValues,
} from '@features/generation/settings';
import { requestWorkflowDocumentLoad } from '@features/workflow/react';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';

import {
  buildImageRecallSettings,
  getImageRecallMessage,
  getImageRecallTitle,
  isImageRecallKindAvailable,
  type ImageRecallKind,
} from './imageRecall';

/**
 * Open the workflow editor before requesting replacement with the image's embedded graph; false means no editor
 * can consume it.
 */
export const executeLoadImageWorkflow = async ({
  image,
  isProjectActive,
  notifications,
  openWorkflowEditor,
  t,
}: {
  image: GalleryImage;
  /** Re-checked after the fetch: a store-resident request must not land on another project's editor. */
  isProjectActive: () => boolean;
  notifications: Pick<WorkbenchCommands['notifications'], 'add'>;
  openWorkflowEditor: () => boolean;
  t: TFunction;
}): Promise<void> => {
  const owner = captureAccountScope();

  try {
    const { workflow } = await galleryImages.workflow(image.imageName, owner.signal);

    assertAccountScopeCurrent(owner);

    if (!workflow) {
      notifications.add({ kind: 'info', title: t('widgets.gallery.itemActions.loadWorkflow.missing') });
      return;
    }

    if (!isProjectActive()) {
      return;
    }

    if (!openWorkflowEditor()) {
      throw new Error(t('widgets.gallery.itemActions.loadWorkflow.editorUnavailable'));
    }

    const raw: unknown = JSON.parse(workflow);
    const name =
      typeof raw === 'object' && raw !== null && typeof (raw as { name?: unknown }).name === 'string'
        ? (raw as { name: string }).name
        : '';

    requestWorkflowDocumentLoad(
      raw,
      t('widgets.gallery.itemActions.loadWorkflow.loadedLabel', { name: name || image.imageName })
    );
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    notifications.add({
      kind: 'error',
      message: toErrorMessage(error),
      title: t('widgets.gallery.itemActions.loadWorkflow.failed'),
    });
  }
};

const imageMetadataRequests = new Map<string, { owner: AccountScope; promise: Promise<unknown> }>();

registerAccountOwnedResource({
  clear: () => {
    imageMetadataRequests.clear();
  },
  name: 'image-recall-metadata',
});

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const loadImageMetadata = (imageName: string, owner: AccountScope): Promise<unknown> => {
  const cachedRequest = imageMetadataRequests.get(imageName);

  if (cachedRequest?.owner === owner) {
    return cachedRequest.promise;
  }

  const request = galleryImages
    .metadata(imageName, owner.signal)
    .then((metadata) => {
      assertAccountScopeCurrent(owner);

      return metadata;
    })
    .catch((error: unknown) => {
      if (imageMetadataRequests.get(imageName)?.promise === request) {
        imageMetadataRequests.delete(imageName);
      }
      throw error;
    });

  imageMetadataRequests.set(imageName, { owner, promise: request });
  return request;
};

/**
 * Drops reference images whose effective asset is no longer in the gallery,
 * using one bulk lookup. Entries without an asset are dropped as well.
 */
export const filterAvailableReferenceImages = async (
  referenceImages: GenerateReferenceImage[],
  signal: AbortSignal
): Promise<GenerateReferenceImage[]> => {
  const effectiveNames = referenceImages.flatMap((referenceImage) =>
    referenceImage.config.image ? [getEffectiveReferenceImage(referenceImage.config.image).image_name] : []
  );
  const availableNames = new Set(
    (await galleryImages.resolveMany([...new Set(effectiveNames)], signal)).map(
      (availableImage) => availableImage.imageName
    )
  );

  return referenceImages.filter((referenceImage) => {
    const referenceAsset = referenceImage.config.image;
    return referenceAsset && availableNames.has(getEffectiveReferenceImage(referenceAsset).image_name);
  });
};

/** Why `getCurrentGenerateValues` answered null: no supported model, or no table to initialise from. */
export const getMissingGenerateValuesMessage = (t: TFunction): string =>
  hasArchitectureCapabilities()
    ? t('widgets.generate.selectSupportedModelFirst')
    : t('widgets.generate.capabilitiesUnavailableForSetup');

export const getCurrentGenerateValues = ({
  generateValues,
  supportedModels,
}: {
  generateValues: Record<string, unknown>;
  supportedModels: GenerateModelConfig[];
}) => {
  const normalizedValues = normalizeGenerateWidgetValues(generateValues);

  if (normalizedValues) {
    return normalizedValues;
  }

  // A first set of values is synthesised from architecture policy, and every caller persists it.
  if (!hasArchitectureCapabilities()) {
    return null;
  }

  const fallbackModelKey = typeof generateValues.modelKey === 'string' ? generateValues.modelKey : null;
  const fallbackModel = supportedModels.find((model) => model.key === fallbackModelKey) ?? supportedModels[0];

  return fallbackModel
    ? ({
        ...getDefaultGenerateSettings(fallbackModel),
        ...generateValues,
        model: fallbackModel,
        modelKey: fallbackModel.key,
      } as GenerateWidgetValues)
    : null;
};

export const executeImageRecall = async ({
  commands,
  generateValues,
  getGenerateValues,
  image,
  kind,
  models,
  owner: callerOwner,
  projectId,
  t,
}: {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  generateValues: Record<string, unknown>;
  getGenerateValues?: () => Record<string, unknown>;
  image: GalleryImage;
  kind: ImageRecallKind;
  models: ModelConfig[];
  /** Caller-captured identity lifetime; direct synchronous callers may omit it. */
  owner?: AccountScope;
  projectId?: string;
  t: TFunction;
}): Promise<boolean> => {
  const owner = callerOwner ?? captureAccountScope();
  if (!isAccountScopeCurrent(owner)) {
    return false;
  }

  const supportedModels = models.filter(isSupportedGenerateModel);
  const vaeModels = models.filter(isVaeModelConfig).map((model) => model as VaeModelConfig);
  const currentGenerateValues = getCurrentGenerateValues({
    generateValues: getGenerateValues?.() ?? generateValues,
    supportedModels,
  });

  try {
    if (!currentGenerateValues) {
      commands.notifications.add({
        kind: 'info',
        message: getMissingGenerateValuesMessage(t),
        title: 'Cannot recall image data',
      });
      return false;
    }

    if (!isImageRecallKindAvailable(kind)) {
      commands.notifications.add({
        kind: 'info',
        message: t('widgets.generate.capabilitiesUnavailableForRecall'),
        title: 'Cannot recall image data',
      });
      return false;
    }

    const metadata = kind === 'dimensions' ? null : await loadImageMetadata(image.imageName, owner);

    assertAccountScopeCurrent(owner);
    const result = buildImageRecallSettings({
      currentValues: currentGenerateValues,
      image,
      kind,
      metadata,
      models,
      supportedModels,
      vaeModels,
    });

    if (!result) {
      commands.notifications.add({
        kind: 'info',
        message: 'This image does not include supported Generate metadata.',
        title: 'No recallable image data',
      });
      return false;
    }

    if (result.fields.includes('referenceImages')) {
      result.values.referenceImages = await filterAvailableReferenceImages(result.values.referenceImages, owner.signal);
      assertAccountScopeCurrent(owner);

      if (result.values.referenceImages.length === 0) {
        result.fields = result.fields.filter((field) => field !== 'referenceImages');
      }
    }

    if (result.fields.length === 0) {
      commands.notifications.add({
        kind: 'info',
        message: 'This image does not include supported Generate metadata.',
        title: 'No recallable image data',
      });
      return false;
    }

    commands.generation.setSettings(result.values, projectId);
    commands.notifications.add({
      kind: 'success',
      message: getImageRecallMessage(result.fields),
      title: getImageRecallTitle(kind),
    });
    return true;
  } catch (error: unknown) {
    if (!isAccountScopeCurrent(owner)) {
      return false;
    }

    commands.notifications.reportError({
      area: 'image-recall',
      message: toErrorMessage(error),
      namespace: 'generation',
      projectId,
    });
    return false;
  }
};
