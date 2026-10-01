import type { CanvasLayerContract, CanvasNodeContract } from '@workbench/canvas-engine/contracts';

import { getDocumentIndex, getDocumentLeaves, haveSameStructure } from '@workbench/canvas-engine/api';
import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createInpaintMaskLayer } from '@workbench/widgets/layers/layerOps';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { describe, expect, it } from 'vitest';

import { createCanvasEngine } from './createCanvasEngine';

const raster = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const setup = (layers: CanvasNodeContract[], selectedLayerId: string) => {
  const store = createWorkbenchStore();
  const projectId = store.getState().activeProjectId;
  store.commands.canvas.apply(projectId, {
    document: { ...createEmptyCanvasDocument(), stacks: stacksFrom(layers), selectedLayerId },
    type: 'replaceCanvasDocument',
  });
  const engine = createCanvasEngine({
    ensureProjectOnServer: () => Promise.resolve(),
    backend: createTestStubRasterBackend(),
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });
  const document = () => store.queries.getProject(projectId)!.canvas.document;
  return { document, engine, original: document() };
};

describe('layer restore through history on an interleaved document', () => {
  it('undoes a delete and then the duplicate that preceded it', async () => {
    const { document, engine, original } = setup(
      [raster('r0'), createInpaintMaskLayer('m1', 'm1'), raster('r1')],
      'r0'
    );
    const duplicated = await engine.layers.duplicateLayers(['r0']);
    expect(duplicated.status).toBe('duplicated');

    const r0 = getDocumentLeaves(document()).find((layer) => layer.id === 'r0')!;
    const remove = engine.document.model()!.prepare({ ids: ['r0'], type: 'remove' });
    if (remove.status !== 'prepared') {
      throw new Error('expected a prepared removal');
    }
    expect(engine.layers.commitPrepared('Delete', remove.edit).status).toBe('committed');
    expect(getDocumentLeaves(document()).some((layer) => layer.id === 'r0')).toBe(false);

    expect(await engine.history.undo()).toBe('applied');
    expect(getDocumentLeaves(document()).find((layer) => layer.id === 'r0')).toEqual(r0);

    expect(await engine.history.undo()).toBe('applied');
    expect(haveSameStructure(document().stacks, original.stacks)).toBe(true);
    expect(document().selectedLayerId).toBe('r0');
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('restores a multi-layer delete between the surviving neighbours before undoing the duplicate', async () => {
    const { document, engine, original } = setup(
      [raster('r0'), createInpaintMaskLayer('m1', 'm1'), raster('r1')],
      'r0'
    );
    const duplicated = await engine.layers.duplicateLayers(['r0']);
    if (duplicated.status !== 'duplicated') {
      throw new Error('expected a duplicate');
    }
    const duplicateId = duplicated.duplicateIds[0]!;

    const remove = engine.document.model()!.prepare({ ids: [duplicateId, 'r0'], type: 'remove' });
    if (remove.status !== 'prepared') {
      throw new Error('expected a prepared removal');
    }
    expect(engine.layers.commitPrepared('Delete', remove.edit).status).toBe('committed');
    expect(getDocumentLeaves(document()).map((layer) => layer.id)).toEqual(['m1', 'r1']);

    expect(await engine.history.undo()).toBe('applied');
    expect(
      getDocumentLeaves(document())
        .filter((layer) => layer.type === 'raster')
        .map((layer) => layer.id)
    ).toEqual([duplicateId, 'r0', 'r1']);

    expect(await engine.history.undo()).toBe('applied');
    expect(haveSameStructure(document().stacks, original.stacks)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('duplicates a group as a whole subtree with fresh ids and removes it again on undo', async () => {
    const { document, engine, original } = setup(
      [raster('r0'), groupContract('g', [raster('r1'), groupContract('h', [raster('r2')])]), raster('r3')],
      'g'
    );
    const outline = () =>
      getDocumentIndex(document()).nodes.map((entry) => `${'  '.repeat(entry.path.length)}${entry.node.id}`);

    const duplicated = await engine.layers.duplicateLayers(['g', 'r2']);
    if (duplicated.status !== 'duplicated') {
      throw new Error(`expected a duplicate, got ${duplicated.status}`);
    }
    expect(duplicated.duplicateIds).toHaveLength(1);
    const [copyId] = duplicated.duplicateIds;
    expect(duplicated.selectedLayerId).toBe(copyId);
    expect(document().selectedLayerId).toBe(copyId);
    const copy = getDocumentIndex(document()).byId.get(copyId!)!;
    expect(copy).toMatchObject({ parentId: null, siblingIndex: 1 });
    expect(copy.node).toMatchObject({ name: 'g copy', type: 'group' });
    const copiedIds = getDocumentIndex(document())
      .nodes.filter((entry) => entry.path.includes(copyId!))
      .map((entry) => entry.node.id);
    expect(copiedIds).toHaveLength(3);
    expect(new Set([...copiedIds, ...getDocumentLeaves(original).map((layer) => layer.id)]).size).toBe(7);
    expect(outline()).toHaveLength(10);

    expect(await engine.history.undo()).toBe('applied');
    expect(haveSameStructure(document().stacks, original.stacks)).toBe(true);
    expect(document().selectedLayerId).toBe('g');
    expect(await engine.history.redo()).toBe('applied');
    expect(getDocumentIndex(document()).byId.get(copyId!)?.node).toMatchObject({ name: 'g copy' });
    engine.lifecycle.dispose();
  });
});
