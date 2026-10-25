import type { AccountScope } from '@platform/state/accountLifecycle';
import type {
  AdjustmentRefusal,
  CanvasDocumentCapability,
  CanvasDocumentContractV3,
  CanvasExportCapability,
} from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { Project } from '@workbench/projectContracts';

import { galleryTransfers, type GalleryImage } from '@features/gallery';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getDocumentLayer, lookupDocumentLeaf } from '@workbench/canvas-engine/api';
import { getControlImageState } from '@workbench/canvasLayerContent';

import type { ControlLayerIntentResult } from './operationTypes';

import { uploadCanvasImage } from './backend/canvasImages';
import { exportCanvasComposite, type CanvasCompositeExportEngine } from './exportCanvasComposite';
import {
  importGalleryImagesToCanvas,
  type GalleryImportEngine,
  type ImportGalleryImagesResult,
} from './importGalleryImages';
import { getCanvasOperations } from './operationAccess';

/** Where a control layer's image came from, when this canvas placed it: a file (by name) or the frame. */
export type ControlSourceOrigin = { kind: 'file'; name: string } | { kind: 'frame' };

export interface ControlSourceSnapshot {
  /** Control layers with an image upload on its way; another upload to one is refused until it lands. */
  readonly uploading: ReadonlySet<string>;
  /** Origins of images placed from a file or the frame, by image name; others are described by their size. */
  readonly origins: Readonly<Record<string, ControlSourceOrigin>>;
}

const EMPTY_SOURCE_SNAPSHOT: ControlSourceSnapshot = { origins: {}, uploading: new Set() };

interface ControlSourceState {
  snapshot: ControlSourceSnapshot;
  readonly listeners: Set<() => void>;
}

// Kept per engine, as its operations are: an account or project switch replaces the engine, and this state with it.
const sourceStates = new WeakMap<object, ControlSourceState>();

const sourceStateOf = (engine: object): ControlSourceState => {
  let state = sourceStates.get(engine);
  if (!state) {
    state = { listeners: new Set(), snapshot: EMPTY_SOURCE_SNAPSHOT };
    sourceStates.set(engine, state);
  }
  return state;
};

const updateSourceState = (engine: object, next: (snapshot: ControlSourceSnapshot) => ControlSourceSnapshot) => {
  const state = sourceStateOf(engine);
  state.snapshot = next(state.snapshot);
  for (const listener of state.listeners) {
    listener();
  }
};

export const getControlSourceSnapshot = (engine: object): ControlSourceSnapshot =>
  sourceStates.get(engine)?.snapshot ?? EMPTY_SOURCE_SNAPSHOT;

export const subscribeControlSourceSnapshot = (engine: object, listener: () => void): (() => void) => {
  const { listeners } = sourceStateOf(engine);
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};

const recordOrigin = (engine: object, imageName: string, origin: ControlSourceOrigin) =>
  updateSourceState(engine, (snapshot) => ({ ...snapshot, origins: { ...snapshot.origins, [imageName]: origin } }));

const setUploading = (engine: object, layerId: string, uploading: boolean) =>
  updateSourceState(engine, (snapshot) => {
    const next = new Set(snapshot.uploading);
    if (uploading) {
      next.add(layerId);
    } else {
      next.delete(layerId);
    }
    return { ...snapshot, uploading: next };
  });

export type UploadControlImageResult =
  | { status: 'applied'; result: ControlLayerIntentResult; imageName: string }
  /** Another upload to the layer is still on its way. */
  | { status: 'uploading' }
  /** The layer was empty when the upload started and is not now (drawn on, or no longer a control layer). */
  | { status: 'changed' }
  | { status: 'stale' }
  | { status: 'failed'; message: string };

/**
 * Uploads an image file and makes it a control layer's image at the bbox origin (`replace-image`, which detects its
 * map), recording the file's name as the image's origin. The Properties Upload button and a file dropped on the
 * canvas share it. One upload per layer at a time; `onUploaded` runs once the file is stored, for the caller to
 * refresh the gallery it now appears in. A layer that was empty when the upload started must still be, so strokes
 * drawn meanwhile are never replaced; replacing an image the layer already had is what was asked for. The upload
 * stays with the account that started it.
 */
export const uploadControlImageFile = async (options: {
  engine: {
    readonly document: Pick<CanvasDocumentCapability, 'getDocument'>;
    readonly exports: Pick<CanvasExportCapability, 'hasExportableLayerContent'>;
  };
  layerId: string;
  file: File;
  onUploaded?: (owner: AccountScope) => void;
  uploadImage?: typeof galleryTransfers.upload;
}): Promise<UploadControlImageResult> => {
  const { engine, file, layerId, onUploaded, uploadImage = galleryTransfers.upload } = options;
  if (getControlSourceSnapshot(engine).uploading.has(layerId)) {
    return { status: 'uploading' };
  }
  const isEmptyControlLayer = (): boolean => {
    const document = engine.document.getDocument();
    const layer = document ? getDocumentLayer(document, layerId) : null;
    return (
      layer?.type === 'control' &&
      getControlImageState(layer, engine.exports.hasExportableLayerContent(layerId)) === 'empty'
    );
  };
  const wasEmpty = isEmptyControlLayer();
  const owner = captureAccountScope();
  setUploading(engine, layerId, true);
  try {
    const uploaded = await uploadImage(file, 'none', { signal: owner.signal });
    assertAccountScopeCurrent(owner);
    onUploaded?.(owner);
    if (wasEmpty && !isEmptyControlLayer()) {
      return { status: 'changed' };
    }
    const bbox = engine.document.getDocument()?.bbox;
    const result = getCanvasOperations(engine).applyControlLayerIntent(layerId, {
      image: { height: uploaded.height, imageName: uploaded.imageName, width: uploaded.width },
      kind: 'replace-image',
      // New images land at the bbox origin, as every other canvas import does.
      transform: bbox ? { rotation: 0, scaleX: 1, scaleY: 1, x: bbox.x, y: bbox.y } : undefined,
    });
    if (result === 'updated') {
      recordOrigin(engine, uploaded.imageName, { kind: 'file', name: file.name });
    }
    return { imageName: uploaded.imageName, result, status: 'applied' };
  } catch (error) {
    return isAccountScopeCurrent(owner)
      ? { message: error instanceof Error ? error.message : String(error), status: 'failed' }
      : { status: 'stale' };
  } finally {
    setUploading(engine, layerId, false);
  }
};

/** A notice for the user: locale keys for the title and message, or a message as the server gave it. */
export interface ControlSourceNotice {
  kind: 'info' | 'error';
  titleKey: string;
  messageKey?: string;
  message?: string;
}

/** Why a control layer intent did not land; `updated`, `unchanged` and `missing` (the layer went away) say nothing. */
export const getControlIntentNotice = (result: ControlLayerIntentResult): ControlSourceNotice | null => {
  switch (result) {
    case 'blocked':
      return { kind: 'info', titleKey: 'widgets.layers.actions.busy' };
    case 'locked':
      return { kind: 'info', titleKey: 'widgets.layers.actions.locked' };
    case 'no-detection-input':
      return { kind: 'info', titleKey: 'widgets.layers.control.detectionNoInput' };
    case 'over-budget':
      return {
        kind: 'error',
        messageKey: 'widgets.layers.actions.overBudget',
        titleKey: 'widgets.layers.actions.actionFailed',
      };
    case 'failed':
      return {
        kind: 'error',
        messageKey: 'widgets.layers.actions.operationFailed',
        titleKey: 'widgets.layers.actions.actionFailed',
      };
    default:
      return null;
  }
};

export const getControlUploadNotice = (outcome: UploadControlImageResult): ControlSourceNotice | null => {
  switch (outcome.status) {
    case 'applied':
      return getControlIntentNotice(outcome.result);
    case 'uploading':
      return { kind: 'info', titleKey: 'widgets.layers.control.source.uploadPending' };
    case 'changed':
      return { kind: 'info', titleKey: 'widgets.layers.control.source.changedDuringUpload' };
    case 'failed':
      return { kind: 'error', message: outcome.message, titleKey: 'widgets.layers.control.source.uploadFailed' };
    case 'stale':
      // The account changed on the way; the result belongs to nobody here.
      return null;
  }
};

export type ControlFileDropResolution =
  | { status: 'accepted'; layerId: string }
  | { status: 'refused'; notice: ControlSourceNotice };

/**
 * Whether files dragged onto the canvas can fill the selected control layer: exactly one image file, the canvas
 * accepting edits, and the selected layer a control layer that is not locked (by itself or a group) and still empty.
 * `files` are the drag's file items while it hovers (their `type` is readable then, to set the drop effect), and its
 * files once dropped.
 */
export const resolveControlFileDrop = (input: {
  document: CanvasDocumentContractV3;
  engine: { readonly exports: Pick<CanvasExportCapability, 'hasExportableLayerContent'> };
  files: readonly { readonly type: string }[];
  isCanvasLocked: boolean;
}): ControlFileDropResolution => {
  const { document, engine, files, isCanvasLocked } = input;
  const refuse = (titleKey: string): ControlFileDropResolution => ({
    notice: { kind: 'info', titleKey },
    status: 'refused',
  });
  if (files.length !== 1) {
    return refuse('widgets.canvas.fileDrop.oneFile');
  }
  if (!files[0]!.type.startsWith('image/')) {
    return refuse('widgets.canvas.fileDrop.notImage');
  }
  if (isCanvasLocked) {
    return refuse('widgets.layers.actions.busy');
  }
  const layer = getDocumentLayer(document, document.selectedLayerId);
  if (
    layer?.type !== 'control' ||
    getControlImageState(layer, engine.exports.hasExportableLayerContent(layer.id)) !== 'empty'
  ) {
    return refuse('widgets.canvas.fileDrop.noTarget');
  }
  if (lookupDocumentLeaf(document, layer.id)?.effectiveLocked !== false) {
    return refuse('widgets.layers.actions.locked');
  }
  return { layerId: layer.id, status: 'accepted' };
};

export type FillControlFromBboxResult =
  | { status: 'applied'; result: ControlLayerIntentResult; imageName: string }
  | { status: 'empty' | 'stale' | 'not-ready' | 'over-budget' }
  | AdjustmentRefusal;

/**
 * Uses the visible raster content inside the bbox as a control layer's image: the composite is uploaded as a control
 * image and replaces the layer's image at the bbox origin, which detects its map again. The upload stays with the
 * account that started it; a sign-out on the way reports `stale`.
 */
export const fillControlFromBbox = async (options: {
  engine: CanvasCompositeExportEngine & { readonly projectId: string };
  layerId: string;
  uploadImage?: typeof uploadCanvasImage;
}): Promise<FillControlFromBboxResult> => {
  const { engine, layerId, uploadImage = uploadCanvasImage } = options;
  const owner = captureAccountScope();
  try {
    const exported = await exportCanvasComposite(engine, 'bbox');
    assertAccountScopeCurrent(owner);
    if (exported.status !== 'ok') {
      return exported;
    }
    const uploaded = await uploadImage(exported.blob, {
      fileName: 'control.png',
      imageCategory: 'control',
      projectId: engine.projectId,
      signal: owner.signal,
    });
    assertAccountScopeCurrent(owner);
    const result = getCanvasOperations(engine).applyControlLayerIntent(layerId, {
      image: { height: uploaded.height, imageName: uploaded.imageName, width: uploaded.width },
      kind: 'replace-image',
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: exported.rect.x, y: exported.rect.y },
    });
    if (result === 'updated') {
      recordOrigin(engine, uploaded.imageName, { kind: 'frame' });
    }
    return { imageName: uploaded.imageName, result, status: 'applied' };
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return { status: 'stale' };
    }
    throw error;
  }
};

export type SendGalleryImageAsControlResult =
  | { status: 'filled'; layerId: string; layerName: string }
  | { status: 'created'; layerId: string }
  | Exclude<ImportGalleryImagesResult, { status: 'imported' }>;

type SendAsControlEngine = GalleryImportEngine & {
  readonly exports: Pick<CanvasExportCapability, 'hasExportableLayerContent'>;
};

/**
 * Uses a gallery image as a control layer's image. The selected control layer takes it when that layer has no image
 * yet and the project is the one on the live canvas (`filled`, detected like any replaced image); otherwise a new
 * control layer holds it (`created`), as the canvas import makes it. Every refusal of either path is reported, and a
 * `project` older than the latest document is `stale-document`.
 */
export const sendGalleryImageAsControl = async (options: {
  applyCanvasMutation: (projectId: string, mutation: CanvasProjectMutation) => boolean | void;
  engine: SendAsControlEngine | null;
  getProject: (projectId: string) => Project | null;
  image: GalleryImage;
  isActiveProject: (projectId: string) => boolean;
  project: Project;
}): Promise<SendGalleryImageAsControlResult> => {
  const { engine, getProject, image, isActiveProject, project } = options;
  const latest = getProject(project.id);
  if (!latest) {
    return { status: 'stale-project' };
  }
  if (latest.canvas.document !== project.canvas.document) {
    return { status: 'stale-document' };
  }

  const { document } = project.canvas;
  const selected = getDocumentLayer(document, document.selectedLayerId);
  const liveEngine = engine !== null && engine.projectId === project.id && isActiveProject(project.id) ? engine : null;
  if (
    liveEngine &&
    selected?.type === 'control' &&
    // Locked by itself or by a group above it: the image goes to a new layer instead.
    lookupDocumentLeaf(document, selected.id)?.effectiveLocked === false &&
    getControlImageState(selected, liveEngine.exports.hasExportableLayerContent(selected.id)) === 'empty'
  ) {
    const result = getCanvasOperations(liveEngine).applyControlLayerIntent(selected.id, {
      image: { height: image.height, imageName: image.imageName, width: image.width },
      kind: 'replace-image',
      // At the bbox origin, as every other canvas import places an image.
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: document.bbox.x, y: document.bbox.y },
    });
    switch (result) {
      case 'updated':
      case 'unchanged':
      case 'no-detection-input':
        return { layerId: selected.id, layerName: selected.name, status: 'filled' };
      case 'missing':
        return { status: 'stale-document' };
      case 'blocked':
      case 'over-budget':
        return { status: 'blocked' };
      case 'failed':
        throw new Error(`Could not use ${image.imageName} as the image of control layer ${selected.id}.`);
      case 'locked':
        // Locked by a group above it: the image goes to a new layer instead.
        break;
    }
  }

  const imported = await importGalleryImagesToCanvas({ ...options, destination: 'control', images: [image] });
  if (imported.status !== 'imported') {
    return imported;
  }
  const [layerId] = imported.layerIds;
  return layerId === undefined ? { status: 'empty' } : { layerId, status: 'created' };
};
