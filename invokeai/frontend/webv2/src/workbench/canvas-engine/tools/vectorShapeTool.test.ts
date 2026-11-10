import type { PointerInput } from '../types';
import type { ToolContext } from './tool';
import { describe, expect, it, vi } from 'vitest';
import { createEngineStores } from '../engineStores';
import { documentFrom } from '../document-model/documentFixtures.testStub';
import { createVectorLayer } from '../document/layerFactories';
import { createViewport } from '../viewport';
import { createTestInsertionAnchorCapture } from '../document/insertionAnchors.testStub';
import { createVectorShapeTool } from './vectorShapeTool';

const pointer = (x: number, y: number, timeStamp = 0): PointerInput => ({ buttons: 1, documentPoint: { x, y }, screenPoint: { x, y },
  pressure: 0.5, pointerType: 'mouse', timeStamp, modifiers: { alt: false, ctrl: false, meta: false, shift: false } });
const harness = () => {
  const layer = createVectorLayer({ id: 'v', name: 'Vector' }), doc = documentFrom([layer], layer.id);
  const stores = createEngineStores(); stores.shapeOptions.set({ ...stores.shapeOptions.get(), target: 'selected', kind: 'rect' });
  let nextId = 0;
  const commit = vi.fn<ToolContext['commitStructural']>(() => ({ status: 'committed' }));
  const ctx = { getDocument: () => doc, stores, viewport: createViewport(), commitStructural: commit,
    createLayerId: () => `path-${++nextId}`, captureInsertionAnchor: createTestInsertionAnchorCapture('p'),
    scheduleFrame: (task: () => void) => { task(); return () => {}; }, invalidate: vi.fn(), updateCursor: vi.fn(), reportEditRefusal: vi.fn(),
  } as unknown as ToolContext;
  return { tool: createVectorShapeTool(), ctx, commit, stores };
};
describe('vector Shapes draft safety', () => {
  it('keeps a refused shape visible, reports the reason and retries it with Enter', () => {
    const h = harness(); h.commit.mockReturnValueOnce({ status: 'over-budget' });
    h.tool.onPointerDown!(h.ctx, pointer(10, 10)); h.tool.onPointerMove!(h.ctx, pointer(100, 50), [pointer(100, 50)]);
    h.tool.onPointerUp!(h.ctx, pointer(100, 50));
    expect(h.tool.hasSession()).toBe(true); expect(h.stores.vectorShapePreview.get()?.points).toHaveLength(4);
    expect(h.ctx.reportEditRefusal).toHaveBeenCalledWith('over-budget');
    h.tool.onKeyCommand!(h.ctx, 'apply');
    expect(h.commit).toHaveBeenCalledTimes(2); expect(h.tool.hasSession()).toBe(false);
  });
  it('continues a polyline between presses until Enter and drops canceled geometry without commits', () => {
    const h = harness(); h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'polygon' });
    for (const [x, y, time] of [[0, 0, 0], [100, 0, 1000], [100, 100, 2000]]) {
      h.tool.onPointerDown!(h.ctx, pointer(x!, y!, time)); h.tool.onPointerUp!(h.ctx, pointer(x!, y!, time));
      expect(h.tool.hasSession()).toBe(true);
    }
    expect(h.commit).not.toHaveBeenCalled(); h.tool.onKeyCommand!(h.ctx, 'apply');
    const mutation = h.commit.mock.calls[0]![1];
    expect(mutation.type).toBe('updateCanvasLayerConfig');
    if (mutation.type !== 'updateCanvasLayerConfig' || mutation.config.layerType !== 'vector') { throw new Error('Expected vector'); }
    expect(mutation.config.paths?.[0]).toMatchObject({ isClosed: false }); expect(mutation.config.paths?.[0]?.points).toHaveLength(3);
    h.tool.onPointerDown!(h.ctx, pointer(20, 20, 3000)); h.tool.onPointerCancel!(h.ctx);
    expect(h.tool.hasSession()).toBe(false); expect(h.commit).toHaveBeenCalledTimes(1);
  });
});
