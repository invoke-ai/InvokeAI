import type { GalleryImage } from '@features/gallery';
import type { ModelConfig } from '@features/models';
import type { CanvasDocumentCapability, CanvasLayerCapability, LayerStackKind } from '@workbench/canvas-engine/api';
import type { CanvasImageRef, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { Project } from '@workbench/projectContracts';

import { getArchitectureCapabilitiesSnapshot } from '@features/generation/runtime';
import {
  calculateNewSize,
  getGenerationDimensions,
  isArchitectureDescribed,
  normalizeGenerateWidgetValues,
} from '@features/generation/settings';
import { getModelsSnapshot } from '@features/models';
import { mapWithConcurrency } from '@platform/core/concurrency';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { captureInsertionAnchor, getDocumentLeaves } from '@workbench/canvas-engine/api';
import { resolveDefaultControlModelForBase } from '@workbench/widgets/layers/controlModelOptions';
import {
  createControlLayer,
  createEmptyPaintLayer,
  createInpaintMaskFromImage,
  createLayerId,
  createRegionalGuidanceFromImage,
  createRegionalGuidanceLayerWithRefImage,
  DEFAULT_INPAINT_MASK_FILL,
  nextControlLayerName,
  nextInpaintMaskName,
  nextRegionalGuidanceFillColor,
  nextRegionalGuidanceName,
} from '@workbench/widgets/layers/layerOps';
import { getSelectedModelBase } from '@workbench/widgets/layers/selectedModel';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { nextLayerName } from '@workbench/workbenchState';

import { uploadCanvasImage } from './backend/canvasImages';

export type GalleryCanvasImportDestination =
  | 'raster'
  | 'control'
  | 'inpaint-mask'
  | 'regional-guidance'
  | 'regional-reference'
  | 'control-resized';

export type ImportGalleryImagesResult =
  | { status: 'imported'; layerIds: string[]; failedImageNames: string[] }
  | { status: 'blocked' | 'capabilities-unavailable' | 'empty' | 'stale-document' | 'stale-project' };

interface BuildLayerContext {
  bbox: Project['canvas']['document']['bbox'];
  defaultControlModel: string | null;
  existingLayers: readonly CanvasLayerContract[];
  modelBase: string | null;
}

type LayerImage = CanvasImageRef | GalleryImage;

const activeImports = new Set<string>();

registerAccountOwnedResource({
  clear: () => {
    activeImports.clear();
  },
  name: 'canvas-gallery-imports',
});

const imageRef = (image: LayerImage): CanvasImageRef => ({
  height: image.height,
  imageName: image.imageName,
  width: image.width,
});

const isGeneratedImage = (image: LayerImage): image is GalleryImage => 'imageUrl' in image;

const withBboxOrigin = <T extends CanvasLayerContract>(layer: T, bbox: BuildLayerContext['bbox']): T => ({
  ...layer,
  transform: { ...layer.transform, x: bbox.x, y: bbox.y },
});

const buildLayer = (
  image: LayerImage,
  destination: GalleryCanvasImportDestination,
  context: BuildLayerContext
): CanvasLayerContract => {
  const id = createLayerId();
  const names = context.existingLayers.map((layer) => layer.name);
  const ref = imageRef(image);

  switch (destination) {
    case 'raster':
      return withBboxOrigin(
        { ...createEmptyPaintLayer(nextLayerName(names), id), source: { image: ref, type: 'image' } },
        context.bbox
      );
    case 'control':
    case 'control-resized':
      return withBboxOrigin(
        {
          ...createControlLayer(nextControlLayerName(names), id, context.modelBase, context.defaultControlModel),
          source: { image: ref, type: 'image' },
        },
        context.bbox
      );
    case 'inpaint-mask':
      return createInpaintMaskFromImage({
        fill: DEFAULT_INPAINT_MASK_FILL,
        id,
        image: ref,
        name: nextInpaintMaskName(names),
        rect: context.bbox,
      });
    case 'regional-guidance': {
      const regionalGuidanceCount = context.existingLayers.filter((layer) => layer.type === 'regional_guidance').length;
      return createRegionalGuidanceFromImage({
        fill: { color: nextRegionalGuidanceFillColor(regionalGuidanceCount), style: 'solid' },
        id,
        image: ref,
        name: nextRegionalGuidanceName(names),
        rect: context.bbox,
      });
    }
    case 'regional-reference': {
      const regionalGuidanceCount = context.existingLayers.filter((layer) => layer.type === 'regional_guidance').length;
      const layer = createRegionalGuidanceLayerWithRefImage(
        nextRegionalGuidanceName(names),
        regionalGuidanceCount,
        context.modelBase,
        id
      );
      const referenceImage = layer.referenceImages[0];
      if (!referenceImage) {
        throw new Error('Regional reference factory did not create a reference image');
      }
      if (!isGeneratedImage(image)) {
        throw new Error('Regional reference imports require a gallery image');
      }
      return {
        ...layer,
        referenceImages: [{ ...referenceImage, config: { ...referenceImage.config, image } }],
      };
    }
  }
};

const buildLayers = (
  images: readonly LayerImage[],
  destination: GalleryCanvasImportDestination,
  project: Project,
  models: readonly ModelConfig[]
): CanvasLayerContract[] => {
  const layers: CanvasLayerContract[] = [];
  const modelBase = getSelectedModelBase(project);
  const defaultControlModel = resolveDefaultControlModelForBase(models, modelBase);
  for (const image of images) {
    layers.push(
      buildLayer(image, destination, {
        bbox: project.canvas.document.bbox,
        defaultControlModel,
        existingLayers: [...getDocumentLeaves(project.canvas.document), ...layers],
        modelBase,
      })
    );
  }
  return layers;
};

type ResizeResult =
  | { status: 'fulfilled'; image: LayerImage }
  | { status: 'rejected'; imageName: string; reason: unknown };

const resizeImages = async (
  images: readonly GalleryImage[],
  project: Project,
  fetchImage: typeof fetch,
  uploadImage: typeof uploadCanvasImage,
  signal: AbortSignal
): Promise<{ images: LayerImage[]; failedImageNames: string[] }> => {
  const generateValues = normalizeGenerateWidgetValues(getProjectWidgetValues(project, 'generate'));
  const dimensions = getGenerationDimensions(generateValues?.model);
  const results = await mapWithConcurrency(images, 4, async (image): Promise<ResizeResult> => {
    try {
      const response = await fetchImage(image.imageUrl, { signal });

      signal.throwIfAborted();
      if (!response.ok) {
        throw new Error(`Failed to fetch ${image.imageName}: ${response.status} ${response.statusText}`);
      }
      const blob = await response.blob();

      signal.throwIfAborted();
      const resizeTo = calculateNewSize(image.width / image.height, dimensions.optimal ** 2, dimensions.grid);
      const uploaded = await uploadImage(blob, {
        fileName: image.imageName,
        imageCategory: 'other',
        isIntermediate: false,
        resizeTo,
        signal,
      });

      signal.throwIfAborted();
      return { image: uploaded, status: 'fulfilled' };
    } catch (reason) {
      return { imageName: image.imageName, reason, status: 'rejected' };
    }
  });
  const successful = results.filter(
    (result): result is Extract<ResizeResult, { status: 'fulfilled' }> => result.status === 'fulfilled'
  );
  const failed = results.filter(
    (result): result is Extract<ResizeResult, { status: 'rejected' }> => result.status === 'rejected'
  );
  if (successful.length === 0 && failed.length > 0) {
    throw new AggregateError(
      failed.map((result) => result.reason),
      `Failed to resize ${String(failed.length)} gallery image${failed.length === 1 ? '' : 's'}`
    );
  }
  return {
    failedImageNames: failed.map((result) => result.imageName),
    images: successful.map((result) => result.image),
  };
};

type GalleryImportEngine = {
  readonly projectId: string;
  readonly document: Pick<CanvasDocumentCapability, 'captureInsertionAnchor'>;
  readonly layers: CanvasLayerCapability;
};

export const importGalleryImagesToCanvas = async (options: {
  applyCanvasMutation: (projectId: string, mutation: CanvasProjectMutation) => boolean | void;
  destination: GalleryCanvasImportDestination;
  engine: GalleryImportEngine | null;
  getProject: (projectId: string) => Project | null;
  isActiveProject: (projectId: string) => boolean;
  images: readonly GalleryImage[];
  project: Project;
  fetchImage?: typeof fetch;
  /** Model list used to pick a default control model; defaults to the loaded models snapshot. */
  models?: readonly ModelConfig[];
  uploadImage?: typeof uploadCanvasImage;
}): Promise<ImportGalleryImagesResult> => {
  const owner = captureAccountScope();
  const {
    applyCanvasMutation,
    destination,
    engine,
    fetchImage = globalThis.fetch,
    getProject,
    images,
    isActiveProject,
    models = getModelsSnapshot().models,
    project,
    uploadImage = uploadCanvasImage,
  } = options;
  if (images.length === 0) {
    return { status: 'empty' };
  }
  const importKey = `${owner.epoch}:${project.id}`;

  if (activeImports.has(importKey)) {
    return { status: 'blocked' };
  }
  activeImports.add(importKey);

  try {
    const capturedDocument = project.canvas.document;
    const matchingProjectEngine = engine !== null && engine.projectId === project.id ? engine : null;
    if (matchingProjectEngine && isActiveProject(project.id) && !matchingProjectEngine.layers.canCommitStructural()) {
      return { status: 'blocked' };
    }

    let layerImages: readonly LayerImage[] = images;
    let failedImageNames: string[] = [];
    if (destination === 'control-resized') {
      // Refuse resize uploads without policy for the selected architecture: fallback size/grid would persist an
      // incorrectly sized asset that cannot be re-derived later.
      const targetModel = normalizeGenerateWidgetValues(getProjectWidgetValues(project, 'generate'))?.model;
      if (
        getArchitectureCapabilitiesSnapshot().revision === 0 ||
        (targetModel !== undefined && !isArchitectureDescribed(targetModel))
      ) {
        return { status: 'capabilities-unavailable' };
      }

      const resized = await resizeImages(images, project, fetchImage, uploadImage, owner.signal);

      assertAccountScopeCurrent(owner);
      layerImages = resized.images;
      failedImageNames = resized.failedImageNames;
    }

    const layers = buildLayers(layerImages, destination, project, models);
    const previousSelectedLayerId = capturedDocument.selectedLayerId;
    const anchorFor = (stack: LayerStackKind) =>
      matchingProjectEngine
        ? matchingProjectEngine.document.captureInsertionAnchor(stack, null)
        : captureInsertionAnchor(capturedDocument.stacks, {
            aboveId: null,
            editRevision: project.canvas.documentRevision,
            projectId: project.id,
            stack,
          });
    const forward: CanvasProjectMutation = {
      add: [...new Set(layers.map((layer) => layer.type))].map((stack) => ({
        anchor: anchorFor(stack),
        nodes: layers.filter((layer) => layer.type === stack),
      })),
      enabledUpdates: [],
      selectedLayerId: layers.at(-1)?.id ?? previousSelectedLayerId,
      type: 'applyCanvasLayerStackMutation',
    };
    const inverse: CanvasProjectMutation = {
      enabledUpdates: [],
      removeIds: layers.map((layer) => layer.id),
      selectedLayerId: previousSelectedLayerId,
      type: 'applyCanvasLayerStackMutation',
    };

    const latestProject = getProject(project.id);

    assertAccountScopeCurrent(owner);
    if (!latestProject) {
      return { status: 'stale-project' };
    }
    if (latestProject.canvas.document !== capturedDocument) {
      return { status: 'stale-document' };
    }

    if (matchingProjectEngine && isActiveProject(project.id)) {
      const committed = matchingProjectEngine.layers.commitStructural('Import gallery images', forward, inverse);
      if (committed.status !== 'committed') {
        return { status: 'blocked' };
      }
    } else {
      // Without a live canvas session, import through ingestion.
      applyCanvasMutation(project.id, forward);
    }
    return { failedImageNames, layerIds: layers.map((layer) => layer.id), status: 'imported' };
  } finally {
    activeImports.delete(importKey);
  }
};
