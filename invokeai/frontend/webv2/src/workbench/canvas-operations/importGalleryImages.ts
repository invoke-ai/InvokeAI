import type { CanvasLayerCapability } from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { GalleryImage } from '@workbench/gallery/api';
import type { CanvasImageRef, CanvasLayerContract, Project, WorkbenchState } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { Dispatch } from 'react';

import { getGenerationDimensions } from '@workbench/generation/baseGenerationPolicies';
import { calculateNewSize, normalizeGenerateWidgetValues } from '@workbench/generation/settings';
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
  | { status: 'blocked' | 'empty' | 'stale-document' | 'stale-project' };

interface BuildLayerContext {
  bbox: Project['canvas']['document']['bbox'];
  existingLayers: readonly CanvasLayerContract[];
  modelBase: string | null;
}

type LayerImage = CanvasImageRef | GalleryImage;

const activeImports = new Set<string>();

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
          ...createControlLayer(nextControlLayerName(names), id, context.modelBase),
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
  project: Project
): CanvasLayerContract[] => {
  const layers: CanvasLayerContract[] = [];
  const modelBase = getSelectedModelBase(project);
  for (const image of images) {
    layers.push(
      buildLayer(image, destination, {
        bbox: project.canvas.document.bbox,
        existingLayers: [...project.canvas.document.layers, ...layers],
        modelBase,
      })
    );
  }
  return layers;
};

const mapWithConcurrency = async <T, R>(
  items: readonly T[],
  concurrency: number,
  mapper: (item: T, index: number) => Promise<R>
): Promise<R[]> => {
  const results: R[] = [];
  let nextIndex = 0;
  const worker = async (): Promise<void> => {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await mapper(items[index]!, index);
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return results;
};

type ResizeResult =
  | { status: 'fulfilled'; image: LayerImage }
  | { status: 'rejected'; imageName: string; reason: unknown };

const resizeImages = async (
  images: readonly GalleryImage[],
  project: Project,
  fetchImage: typeof fetch,
  uploadImage: typeof uploadCanvasImage
): Promise<{ images: LayerImage[]; failedImageNames: string[] }> => {
  const generateValues = normalizeGenerateWidgetValues(getProjectWidgetValues(project, 'generate'));
  const dimensions = getGenerationDimensions(generateValues?.model);
  const results = await mapWithConcurrency(images, 4, async (image): Promise<ResizeResult> => {
    try {
      const response = await fetchImage(image.imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch ${image.imageName}: ${response.status} ${response.statusText}`);
      }
      const blob = await response.blob();
      const resizeTo = calculateNewSize(image.width / image.height, dimensions.optimal ** 2, dimensions.grid);
      const uploaded = await uploadImage(blob, {
        fileName: image.imageName,
        imageCategory: 'other',
        isIntermediate: false,
        resizeTo,
      });
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

type GalleryImportEngine = { readonly projectId: string; readonly layers: CanvasLayerCapability };

export const importGalleryImagesToCanvas = async (options: {
  destination: GalleryCanvasImportDestination;
  dispatch: Dispatch<WorkbenchAction>;
  engine: GalleryImportEngine | null;
  getState: () => WorkbenchState;
  images: readonly GalleryImage[];
  project: Project;
  fetchImage?: typeof fetch;
  uploadImage?: typeof uploadCanvasImage;
}): Promise<ImportGalleryImagesResult> => {
  const {
    destination,
    dispatch,
    engine,
    fetchImage = globalThis.fetch,
    getState,
    images,
    project,
    uploadImage = uploadCanvasImage,
  } = options;
  if (images.length === 0) {
    return { status: 'empty' };
  }
  if (activeImports.has(project.id)) {
    return { status: 'blocked' };
  }
  activeImports.add(project.id);

  try {
    const capturedDocument = project.canvas.document;
    const initialState = getState();
    const matchingProjectEngine = engine !== null && engine.projectId === project.id ? engine : null;
    if (
      matchingProjectEngine &&
      initialState.activeProjectId === project.id &&
      !matchingProjectEngine.layers.canCommitStructural()
    ) {
      return { status: 'blocked' };
    }

    let layerImages: readonly LayerImage[] = images;
    let failedImageNames: string[] = [];
    if (destination === 'control-resized') {
      const resized = await resizeImages(images, project, fetchImage, uploadImage);
      layerImages = resized.images;
      failedImageNames = resized.failedImageNames;
    }

    const layers = buildLayers(layerImages, destination, project);
    const previousSelectedLayerId = capturedDocument.selectedLayerId;
    const forward: CanvasProjectMutation = {
      add: { index: 0, layers },
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

    const latestState = getState();
    const latestProject = latestState.projects.find((candidate) => candidate.id === project.id);
    if (!latestProject) {
      return { status: 'stale-project' };
    }
    if (latestProject.canvas.document !== capturedDocument) {
      return { status: 'stale-document' };
    }

    if (matchingProjectEngine && latestState.activeProjectId === project.id) {
      if (!matchingProjectEngine.layers.commitStructural('Import gallery images', forward, inverse)) {
        return { status: 'blocked' };
      }
    } else {
      dispatch({ mutation: forward, projectId: project.id, type: 'applyCanvasProjectMutation' });
    }
    return { failedImageNames, layerIds: layers.map((layer) => layer.id), status: 'imported' };
  } finally {
    activeImports.delete(project.id);
  }
};
