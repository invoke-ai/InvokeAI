import type { GalleryImage } from '@features/gallery';
import type { CanvasControlLayerContract, CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { Project } from '@workbench/projectContracts';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { createControlLayer } from '@workbench/canvas-engine/api';
import {
  documentFrom,
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { uploadCanvasImage } from './backend/canvasImages';
import type { ControlLayerIntent, ControlLayerIntentResult } from './operationTypes';

import {
  fillControlFromBbox,
  getControlSourceSnapshot,
  resolveControlFileDrop,
  sendGalleryImageAsControl,
  uploadControlImageFile,
} from './controlSourceActions';
import { attachCanvasOperations } from './operationAccess';

const BBOX = { height: 512, width: 768, x: 31, y: 47 };

const engineWith = (composite: unknown) => {
  const intents: ControlLayerIntent[] = [];
  const engine = {
    document: { getDocument: () => ({ bbox: BBOX }) },
    exports: { exportRasterComposite: () => composite },
    lifecycle: { flushPendingUploads: () => Promise.resolve() },
    projectId: 'project-1',
  } as unknown as Parameters<typeof fillControlFromBbox>[0]['engine'];
  attachCanvasOperations(engine, {
    applyControlLayerIntent: (_layerId: string, intent: ControlLayerIntent) => {
      intents.push(intent);
      return 'updated';
    },
  } as never);
  return { engine, intents };
};

afterEach(() => {
  accountLifecycle.invalidate();
});

describe('fillControlFromBbox', () => {
  it('uploads the bbox composite as a control image and places it at the bbox origin', async () => {
    const { engine, intents } = engineWith({ blob: new Blob(['png']), rect: BBOX, status: 'ok' });
    const uploads: Parameters<typeof uploadCanvasImage>[1][] = [];
    const uploadImage: typeof uploadCanvasImage = (_blob, options) => {
      uploads.push(options);
      return Promise.resolve({ height: 512, imageName: 'bbox.png', width: 768 });
    };

    await expect(fillControlFromBbox({ engine, layerId: 'control-1', uploadImage })).resolves.toEqual({
      imageName: 'bbox.png',
      result: 'updated',
      status: 'applied',
    });
    expect(uploads[0]).toMatchObject({ imageCategory: 'control', projectId: 'project-1' });
    expect(intents).toEqual([
      {
        image: { height: 512, imageName: 'bbox.png', width: 768 },
        kind: 'replace-image',
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 31, y: 47 },
      },
    ]);
  });

  it('reports an empty bbox without uploading', async () => {
    const { engine, intents } = engineWith({ status: 'empty' });
    const uploadImage = () => Promise.reject(new Error('should not upload'));

    await expect(fillControlFromBbox({ engine, layerId: 'control-1', uploadImage })).resolves.toEqual({
      status: 'empty',
    });
    expect(intents).toEqual([]);
  });

  it('drops an upload that finishes after the account changed', async () => {
    accountLifecycle.activate('control-source-a');
    const { engine, intents } = engineWith({ blob: new Blob(['png']), rect: BBOX, status: 'ok' });
    const uploadImage: typeof uploadCanvasImage = () => {
      accountLifecycle.activate('control-source-b');
      return Promise.resolve({ height: 1, imageName: 'late.png', width: 1 });
    };

    await expect(fillControlFromBbox({ engine, layerId: 'control-1', uploadImage })).resolves.toEqual({
      status: 'stale',
    });
    expect(intents).toEqual([]);
  });
});

describe('sendGalleryImageAsControl', () => {
  const IMAGE = {
    boardId: 'none',
    height: 160,
    imageCategory: 'general',
    imageName: 'gallery.png',
    imageUrl: '/gallery.png',
    queuedAt: '2026-01-01T00:00:00.000Z',
    sourceQueueItemId: 'queue-1',
    starred: false,
    thumbnailUrl: '/thumb/gallery.png',
    width: 320,
  } as GalleryImage;
  const emptyControl = createControlLayer({ id: 'control-1', name: 'Depth 1' });

  const projectWith = (layers: CanvasLayerContract[], selectedLayerId: string | null): Project => {
    const state = createInitialWorkbenchState();
    const current = state.projects.find((candidate) => candidate.id === state.activeProjectId)!;
    const document = { ...current.canvas.document, bbox: BBOX, selectedLayerId, stacks: stacksFrom(layers) };
    return { ...current, canvas: { ...current.canvas, document } };
  };

  /** A live canvas for `project`; `hasPixels` stands for strokes only the engine's cache holds. */
  const liveEngine = (
    project: Project,
    { hasPixels = false, intentResult = 'updated' as ControlLayerIntentResult } = {}
  ) => {
    const intents: [string, ControlLayerIntent][] = [];
    const commitStructural = vi.fn(() => ({ status: 'committed' as const }));
    const engine = {
      document: { captureInsertionAnchor: createTestInsertionAnchorCapture(project.id) },
      exports: { hasExportableLayerContent: () => hasPixels },
      layers: { canCommitStructural: () => true, commitStructural },
      projectId: project.id,
    } as unknown as Parameters<typeof sendGalleryImageAsControl>[0]['engine'] & object;
    attachCanvasOperations(engine, {
      applyControlLayerIntent: (layerId: string, intent: ControlLayerIntent) => {
        intents.push([layerId, intent]);
        return intentResult;
      },
    } as never);
    return { commitStructural, engine, intents };
  };

  const send = (
    project: Project,
    engine: Parameters<typeof sendGalleryImageAsControl>[0]['engine'],
    latest = project
  ) => {
    const applyCanvasMutation = vi.fn();
    const result = sendGalleryImageAsControl({
      applyCanvasMutation,
      engine,
      getProject: (projectId) => (projectId === latest.id ? latest : null),
      image: IMAGE,
      isActiveProject: (projectId) => projectId === project.id,
      project,
    });
    return { applyCanvasMutation, result };
  };

  it('fills the selected control layer that has no image yet, at the bbox origin', async () => {
    const project = projectWith([emptyControl], 'control-1');
    const { commitStructural, engine, intents } = liveEngine(project);

    await expect(send(project, engine).result).resolves.toEqual({
      layerId: 'control-1',
      layerName: 'Depth 1',
      status: 'filled',
    });
    expect(intents).toEqual([
      [
        'control-1',
        {
          image: { height: 160, imageName: 'gallery.png', width: 320 },
          kind: 'replace-image',
          transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 31, y: 47 },
        },
      ],
    ]);
    expect(commitStructural).not.toHaveBeenCalled();
  });

  it.each([
    ['the selected control layer has an image', [{ ...emptyControl, source: { image: IMAGE, type: 'image' } }], false],
    ['the selected control layer holds strokes not saved yet', [emptyControl], true],
  ] as const)('adds a new control layer when %s', async (_label, layers, hasPixels) => {
    const project = projectWith([...layers] as CanvasControlLayerContract[], 'control-1');
    const { commitStructural, engine, intents } = liveEngine(project, { hasPixels });

    const result = await send(project, engine).result;

    expect(intents).toEqual([]);
    expect(commitStructural).toHaveBeenCalledOnce();
    const [, forward] = commitStructural.mock.calls[0] as unknown as [
      string,
      { add: { nodes: CanvasLayerContract[] }[] },
    ];
    const added = forward.add[0]!.nodes[0]!;
    expect(added).toMatchObject({ source: { image: { imageName: 'gallery.png' }, type: 'image' }, type: 'control' });
    expect(result).toEqual({ layerId: added.id, status: 'created' });
  });

  it('adds a new control layer instead of filling an empty one inside a locked group', async () => {
    const lockedGroup = groupContract('group-1', [emptyControl], { isLocked: true });
    const project = projectWith([lockedGroup as unknown as CanvasLayerContract], 'control-1');
    const { commitStructural, engine, intents } = liveEngine(project);

    const result = await send(project, engine).result;

    expect(intents).toEqual([]);
    expect(commitStructural).toHaveBeenCalledOnce();
    expect(result).toMatchObject({ status: 'created' });
  });

  it('adds a new control layer through ingestion when the project has no live canvas', async () => {
    const project = projectWith([emptyControl], 'control-1');

    const { applyCanvasMutation, result } = send(project, null);

    await expect(result).resolves.toMatchObject({ status: 'created' });
    expect(applyCanvasMutation).toHaveBeenCalledOnce();
  });

  it('refuses a document that changed since the caller read it, and reports a refused fill', async () => {
    const project = projectWith([emptyControl], 'control-1');
    const { engine, intents } = liveEngine(project);
    const latest = { ...project, canvas: { ...project.canvas, document: { ...project.canvas.document } } };

    await expect(send(project, engine, latest).result).resolves.toEqual({ status: 'stale-document' });
    expect(intents).toEqual([]);

    const busy = liveEngine(project, { intentResult: 'blocked' });
    await expect(send(project, busy.engine).result).resolves.toEqual({ status: 'blocked' });
  });
});

describe('uploadControlImageFile', () => {
  const FILE = new File(['png'], 'photo.png', { type: 'image/png' });
  const UPLOADED = { height: 64, imageName: 'photo-upload.png', width: 96 } as GalleryImage;

  /** A canvas holding `layer`; `pixels` stands for strokes only the engine's cache holds, and can change. */
  const canvasWith = (layer: CanvasLayerContract) => {
    const pixels = { drawn: false };
    const intents: ControlLayerIntent[] = [];
    const engine = {
      document: { getDocument: () => ({ ...documentFrom([layer], layer.id), bbox: BBOX }) },
      exports: { hasExportableLayerContent: () => pixels.drawn },
    };
    attachCanvasOperations(engine, {
      applyControlLayerIntent: (_layerId: string, intent: ControlLayerIntent) => {
        intents.push(intent);
        return 'updated';
      },
    } as never);
    return { engine, intents, pixels };
  };

  it('places the file at the bbox origin and names the image after it', async () => {
    const { engine, intents } = canvasWith(createControlLayer({ id: 'control-1', name: 'Depth 1' }));
    const onUploaded = vi.fn();

    await expect(
      uploadControlImageFile({
        engine,
        file: FILE,
        layerId: 'control-1',
        onUploaded,
        uploadImage: () => Promise.resolve(UPLOADED),
      })
    ).resolves.toEqual({ imageName: 'photo-upload.png', result: 'updated', status: 'applied' });
    expect(intents).toEqual([
      {
        image: { height: 64, imageName: 'photo-upload.png', width: 96 },
        kind: 'replace-image',
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 31, y: 47 },
      },
    ]);
    expect(onUploaded).toHaveBeenCalledOnce();
    expect(getControlSourceSnapshot(engine).origins['photo-upload.png']).toEqual({ kind: 'file', name: 'photo.png' });
  });

  it('leaves strokes drawn on an empty layer during the upload in place', async () => {
    const { engine, intents, pixels } = canvasWith(createControlLayer({ id: 'control-1', name: 'Depth 1' }));
    const onUploaded = vi.fn();
    const uploadImage = () => {
      pixels.drawn = true;
      return Promise.resolve(UPLOADED);
    };

    await expect(
      uploadControlImageFile({ engine, file: FILE, layerId: 'control-1', onUploaded, uploadImage })
    ).resolves.toEqual({ status: 'changed' });
    expect(intents).toEqual([]);
    // The image is stored all the same, so the gallery that now lists it is refreshed.
    expect(onUploaded).toHaveBeenCalledOnce();
    expect(getControlSourceSnapshot(engine).uploading.has('control-1')).toBe(false);
  });

  it('replaces the image a layer already had, as the Replace button asks', async () => {
    const withImage = {
      ...createControlLayer({ id: 'control-1', name: 'Depth 1' }),
      source: { image: { height: 8, imageName: 'old.png', width: 8 }, type: 'image' },
    } as CanvasControlLayerContract;
    const { engine, intents } = canvasWith(withImage);

    await expect(
      uploadControlImageFile({ engine, file: FILE, layerId: 'control-1', uploadImage: () => Promise.resolve(UPLOADED) })
    ).resolves.toMatchObject({ status: 'applied' });
    expect(intents).toHaveLength(1);
  });
});

/** Decides the canvas's drop effect while files hover, and whether a drop is used. */
describe('resolveControlFileDrop', () => {
  const emptyControl = createControlLayer({ id: 'control-1', name: 'Depth 1' });
  const IMAGE_ITEM = { type: 'image/png' };
  const resolve = (
    nodes: Parameters<typeof documentFrom>[0],
    selectedLayerId: string,
    { files = [IMAGE_ITEM] as { type: string }[], hasPixels = false, isCanvasLocked = false } = {}
  ) =>
    resolveControlFileDrop({
      document: documentFrom(nodes, selectedLayerId),
      engine: { exports: { hasExportableLayerContent: () => hasPixels } },
      files,
      isCanvasLocked,
    });
  const refusedWith = (titleKey: string) => ({ notice: { kind: 'info', titleKey }, status: 'refused' });

  it('takes one image file onto an empty, unlocked control layer', () => {
    expect(resolve([emptyControl], 'control-1')).toEqual({ layerId: 'control-1', status: 'accepted' });
  });

  it.each([
    [
      'several files',
      [emptyControl],
      {
        files: [IMAGE_ITEM, IMAGE_ITEM],
      },
      'widgets.canvas.fileDrop.oneFile',
    ],
    [
      'a file that is not an image',
      [emptyControl],
      { files: [{ type: 'text/plain' }] },
      'widgets.canvas.fileDrop.notImage',
    ],
    ['a locked canvas', [emptyControl], { isCanvasLocked: true }, 'widgets.layers.actions.busy'],
    ['a layer with strokes', [emptyControl], { hasPixels: true }, 'widgets.canvas.fileDrop.noTarget'],
    [
      'a layer locked by its group',
      [groupContract('group-1', [emptyControl], { isLocked: true })],
      {},
      'widgets.layers.actions.locked',
    ],
  ] as const)('refuses %s', (_label, nodes, options, titleKey) => {
    expect(
      resolve([...nodes], 'control-1', { ...options, files: 'files' in options ? [...options.files] : undefined })
    ).toEqual(refusedWith(titleKey));
  });

  it('refuses a selected layer that is not a control layer', () => {
    expect(resolve([layerContract('raster-1', 'raster')], 'raster-1')).toEqual(
      refusedWith('widgets.canvas.fileDrop.noTarget')
    );
  });
});
