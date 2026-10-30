import type { GalleryImage } from '@features/gallery';
import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateWidgetValues,
  VaeModelConfig,
} from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { QueryClient } from '@tanstack/react-query';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { galleryImages, galleryVideos } from '@features/gallery';
import { fetchGalleryImageMetadata } from '@features/gallery/queries';
import {
  getCompatibleReferenceImages,
  getDefaultGenerateSettings,
  getEffectiveReferenceImage,
  hasArchitectureCapabilities,
  isSupportedGenerateModel,
  isVaeModelConfig,
  normalizeGenerateWidgetValues,
} from '@features/generation/settings';
import {
  getInvocationTemplatesSnapshot,
  refreshInvocationTemplates,
  requestWorkflowDocumentLoad,
} from '@features/workflow/react';
import { serializeWorkflowJson } from '@features/workflow/utility';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  type AccountScope,
} from '@platform/state/accountLifecycle';

import {
  buildImageRecallSettings,
  getImageRecallMessage,
  getMetadataReferenceImages,
  getImageRecallTitle,
  isImageRecallKindAvailable,
  type ImageRecallKind,
  type ImageRecallSkip,
} from './imageRecall';

const getSkippedRefinerMessage = ({ name, reason }: ImageRecallSkip, t: TFunction): string => {
  const values = { name: name ?? t('widgets.gallery.itemActions.recall.unnamedRefiner') };

  switch (reason) {
    case 'ambiguous':
      return t('widgets.gallery.itemActions.recall.skippedRefiner.ambiguous', values);
    case 'invalid':
      return t('widgets.gallery.itemActions.recall.skippedRefiner.invalid', values);
    case 'modelUnavailable':
      return t('widgets.gallery.itemActions.recall.skippedRefiner.modelUnavailable', values);
    default:
      return t('widgets.gallery.itemActions.recall.skippedRefiner.unresolved', values);
  }
};

const getSkippedConceptMessage = ({ name, reason }: ImageRecallSkip, t: TFunction): string => {
  const values = { name: name ?? t('widgets.gallery.itemActions.recall.unnamedConcept') };

  switch (reason) {
    case 'ambiguous':
      return t('widgets.gallery.itemActions.recall.skippedConcept.ambiguous', values);
    case 'duplicate':
      return t('widgets.gallery.itemActions.recall.skippedConcept.duplicate', values);
    case 'incompatible':
      return t('widgets.gallery.itemActions.recall.skippedConcept.incompatible', values);
    case 'invalid':
      return t('widgets.gallery.itemActions.recall.skippedConcept.invalid', values);
    case 'modelUnavailable':
      return t('widgets.gallery.itemActions.recall.skippedConcept.modelUnavailable', values);
    case 'unresolved':
      return t('widgets.gallery.itemActions.recall.skippedConcept.unresolved', values);
  }
};

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export interface WorkflowMediaRef {
  kind: 'image' | 'video';
  /** The image or video name the backend stores the embedded workflow under. */
  name: string;
}

const fetchEmbeddedWorkflow = (media: WorkflowMediaRef, signal: AbortSignal) =>
  media.kind === 'image' ? galleryImages.workflow(media.name, signal) : galleryVideos.workflow(media.name, signal);

/**
 * Load the media's authored workflow, or reconstruct its generation graph, then open the workflow editor. False
 * from `openWorkflowEditor` means no editor can consume the load request.
 */
export const executeLoadMediaWorkflow = async ({
  isProjectActive,
  media,
  notifications,
  openWorkflowEditor,
  t,
}: {
  /** Re-checked after the fetch: a store-resident request must not land on another project's editor. */
  isProjectActive: () => boolean;
  media: WorkflowMediaRef;
  notifications: Pick<WorkbenchCommands['notifications'], 'add'>;
  openWorkflowEditor: () => boolean;
  t: TFunction;
}): Promise<void> => {
  const owner = captureAccountScope();
  const suffix = media.kind === 'video' ? 'Video' : '';

  try {
    const { workflow, graph } = await fetchEmbeddedWorkflow(media, owner.signal);

    assertAccountScopeCurrent(owner);

    if (!isProjectActive()) {
      return;
    }

    if (!workflow && !graph) {
      notifications.add({ kind: 'info', title: t(`widgets.gallery.itemActions.loadWorkflow.missing${suffix}`) });
      return;
    }

    let raw: unknown;

    if (workflow) {
      raw = JSON.parse(workflow);
    } else {
      const { storedGraphToDocument } = await import('@features/workflow/preview');

      assertAccountScopeCurrent(owner);

      if (!isProjectActive()) {
        return;
      }

      if (getInvocationTemplatesSnapshot().status !== 'loaded') {
        await refreshInvocationTemplates();
      }

      assertAccountScopeCurrent(owner);

      if (!isProjectActive()) {
        return;
      }

      const snapshot = getInvocationTemplatesSnapshot();

      if (snapshot.status !== 'loaded') {
        throw new Error(snapshot.error || t('graphPreview.editInEditorFailed'));
      }

      const { document, skippedNodeTypes } = storedGraphToDocument(JSON.parse(graph!), snapshot.templates);

      if (document.nodes.length === 0) {
        throw new Error(t('graphPreview.editInEditorFailed'));
      }

      raw = serializeWorkflowJson(document);

      if (skippedNodeTypes.length > 0) {
        notifications.add({
          kind: 'info',
          title: t('graphPreview.nodesSkipped', { count: skippedNodeTypes.length, types: skippedNodeTypes.join(', ') }),
        });
      }
    }

    if (!isProjectActive()) {
      return;
    }

    if (!openWorkflowEditor()) {
      throw new Error(t('widgets.gallery.itemActions.loadWorkflow.editorUnavailable'));
    }

    const name =
      typeof raw === 'object' && raw !== null && typeof (raw as { name?: unknown }).name === 'string'
        ? (raw as { name: string }).name
        : '';

    requestWorkflowDocumentLoad(
      raw,
      t(`widgets.gallery.itemActions.loadWorkflow.loadedLabel${suffix}`, { name: name || media.name })
    );
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    notifications.add({
      kind: 'error',
      message: toErrorMessage(error),
      title: t(`widgets.gallery.itemActions.loadWorkflow.failed${suffix}`),
    });
  }
};

const getReferenceImageName = (referenceImage: GenerateReferenceImage): string | null =>
  referenceImage.config.image ? getEffectiveReferenceImage(referenceImage.config.image).image_name : null;

/** The names of the references' effective assets the gallery still has, in one bulk lookup; none asks nothing. */
const resolveAvailableReferenceImageNames = async (
  referenceImages: readonly GenerateReferenceImage[],
  signal: AbortSignal
): Promise<ReadonlySet<string>> => {
  const names = [...new Set(referenceImages.flatMap((referenceImage) => getReferenceImageName(referenceImage) ?? []))];

  if (names.length === 0) {
    return new Set();
  }

  return new Set((await galleryImages.resolveMany(names, signal)).map((availableImage) => availableImage.imageName));
};

/** Entries without an asset, or whose effective asset is not among `availableNames`, are dropped. */
const keepAvailableReferenceImages = (
  referenceImages: GenerateReferenceImage[],
  availableNames: ReadonlySet<string>
): GenerateReferenceImage[] =>
  referenceImages.filter((referenceImage) => {
    const name = getReferenceImageName(referenceImage);
    return name !== null && availableNames.has(name);
  });

/** Drops reference images whose effective asset is no longer in the gallery, using one bulk lookup. */
export const filterAvailableReferenceImages = async (
  referenceImages: GenerateReferenceImage[],
  signal: AbortSignal
): Promise<GenerateReferenceImage[]> =>
  keepAvailableReferenceImages(referenceImages, await resolveAvailableReferenceImageNames(referenceImages, signal));

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
  isCanvasFrame,
  kind,
  models,
  owner: callerOwner,
  projectId,
  queryClient,
  t,
}: {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  generateValues: Record<string, unknown>;
  getGenerateValues?: () => Record<string, unknown>;
  image: GalleryImage;
  /** The project's route is Canvas, whose frame the Generate size is. */
  isCanvasFrame?: boolean;
  kind: ImageRecallKind;
  models: ModelConfig[];
  /** Caller-captured identity lifetime; direct synchronous callers may omit it. */
  owner?: AccountScope;
  projectId?: string;
  /** Holds the gallery's shared image metadata entry, which menus and Preview Details may already have read. */
  queryClient: QueryClient;
  t: TFunction;
}): Promise<boolean> => {
  const owner = callerOwner ?? captureAccountScope();
  if (!isAccountScopeCurrent(owner)) {
    return false;
  }

  const supportedModels = models.filter(isSupportedGenerateModel);
  const vaeModels = models.filter(isVaeModelConfig).map((model) => model as VaeModelConfig);

  try {
    if (!isImageRecallKindAvailable(kind)) {
      commands.notifications.add({
        kind: 'info',
        message: t('widgets.generate.capabilitiesUnavailableForRecall'),
        title: t('widgets.gallery.itemActions.recall.cannotRecallImage'),
      });
      return false;
    }

    const readGenerateValues = () =>
      getCurrentGenerateValues({ generateValues: getGenerateValues?.() ?? generateValues, supportedModels });
    const missingValuesNotice = () =>
      commands.notifications.add({
        kind: 'info',
        message: getMissingGenerateValuesMessage(t),
        title: t('widgets.gallery.itemActions.recall.cannotRecallImage'),
      });

    // No reads without values to apply them to: the answer only depends on the model catalog and capabilities,
    // which reading the record does not change.
    if (!readGenerateValues()) {
      missingValuesNotice();
      return false;
    }

    const metadata =
      kind === 'dimensions' ? null : await fetchGalleryImageMetadata(queryClient, owner, image.imageName);
    // Only all/remix recall references. Checked against the record, not the built list, so the check happens before
    // the snapshot below.
    const availableReferenceNames =
      kind === 'all' || kind === 'remix'
        ? await resolveAvailableReferenceImageNames(getMetadataReferenceImages(metadata) ?? [], owner.signal)
        : new Set<string>();

    assertAccountScopeCurrent(owner);
    // Every read is above: nothing yields between this snapshot and the commit, so an edit made while the record
    // or its references loaded stays in the values the recall writes.
    const currentGenerateValues = readGenerateValues();

    if (!currentGenerateValues) {
      missingValuesNotice();
      return false;
    }

    const result = buildImageRecallSettings({
      currentValues: currentGenerateValues,
      image,
      isCanvasFrame,
      kind,
      metadata,
      models,
      supportedModels,
      vaeModels,
    });

    if (!result) {
      commands.notifications.add({
        kind: 'info',
        message: t('widgets.gallery.itemActions.recall.imageMetadataUnsupported'),
        title: t('widgets.gallery.itemActions.recall.nothingRecalled'),
      });
      return false;
    }

    // An empty recalled list clears the current references; there is nothing to check for it.
    if (result.fields.includes('referenceImages') && result.values.referenceImages.length > 0) {
      result.values.referenceImages = keepAvailableReferenceImages(
        result.values.referenceImages,
        availableReferenceNames
      );

      // Every recorded reference was deleted: keep the current ones, fitted to the recalled model as an unrecorded list
      // would be, rather than clearing them as if the record listed none.
      if (result.values.referenceImages.length === 0) {
        result.values.referenceImages = getCompatibleReferenceImages(
          currentGenerateValues.referenceImages,
          result.values.model,
          models
        );
        result.fields = result.fields.filter((field) => field !== 'referenceImages');
      }
    }

    const skippedConcepts = result.skipped.filter((skipped) => skipped.subject === 'concept');
    const skippedRefiner = result.skipped.find((skipped) => skipped.subject === 'refiner');
    const skipMessages = [
      skippedConcepts.length > 0
        ? t('widgets.gallery.itemActions.recall.conceptsNotRestored', {
            concepts: skippedConcepts.map((skipped) => getSkippedConceptMessage(skipped, t)).join('; '),
          })
        : null,
      skippedRefiner ? getSkippedRefinerMessage(skippedRefiner, t) : null,
    ].filter((message) => message !== null);
    const skipMessage = skipMessages.length > 0 ? skipMessages.join(' ') : null;

    if (result.fields.length === 0) {
      commands.notifications.add({
        kind: 'info',
        message: skipMessage ?? t('widgets.gallery.itemActions.recall.imageMetadataUnsupported'),
        title: t('widgets.gallery.itemActions.recall.nothingRecalled'),
      });
      return false;
    }

    commands.generation.setSettings(result.values, projectId);
    commands.notifications.add({
      kind: 'success',
      message: getImageRecallMessage(result.fields, t),
      title: getImageRecallTitle(kind, t),
    });
    if (skipMessage) {
      commands.notifications.add({
        kind: 'info',
        message: skipMessage,
        title: t('widgets.gallery.itemActions.recall.partiallyRecalled'),
      });
    }
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
