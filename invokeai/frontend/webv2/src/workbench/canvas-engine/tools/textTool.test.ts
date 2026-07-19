import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { describe, expect, it, vi } from 'vitest';

import { createTextTool } from './textTool';

const textLayer = (over: Partial<CanvasLayerContract> = {}): CanvasLayerContract =>
  ({
    blendMode: 'normal',
    id: 'text-existing',
    isEnabled: true,
    isLocked: false,
    name: 'Text',
    opacity: 1,
    source: {
      align: 'left',
      color: '#000000',
      content: 'hello',
      fontFamily: 'Inter',
      fontSize: 20,
      fontWeight: 400,
      lineHeight: 1.2,
      type: 'text',
    },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
    ...over,
  }) as CanvasLayerContract;

const makeDoc = (over: Partial<CanvasDocumentContractV2> = {}): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 96, width: 96, x: 0, y: 0 },
  height: 512,
  layers: [],
  selectedLayerId: null,
  version: 2,
  width: 512,
  ...over,
});

const pointer = (x: number, y: number, buttons = 1): PointerInput => ({
  buttons,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: 0,
});

const createHarness = (doc: CanvasDocumentContractV2) => {
  const stores = createEngineStores();
  const openTextCreate = vi.fn<(point: Vec2) => void>();
  const openTextEdit = vi.fn<(layerId: string) => void>();
  const cancelTextEdit = vi.fn<() => void>();
  const ctx = {
    cancelTextEdit,
    getDocument: () => doc,
    // No layer cache in these tests, so hit-testing falls back to the pure
    // estimateTextExtent (a `get` that always misses).
    layers: { get: () => undefined },
    openTextCreate,
    openTextEdit,
    stores,
  } as unknown as ToolContext;
  return { cancelTextEdit, ctx, openTextCreate, openTextEdit, stores };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);

describe('text tool: click behaviour', () => {
  it('opens a create-mode session at the click point on empty area', () => {
    const h = createHarness(makeDoc());
    const tool = createTextTool();

    down(tool, h.ctx, pointer(40, 60));

    expect(h.openTextCreate).toHaveBeenCalledTimes(1);
    expect(h.openTextCreate).toHaveBeenCalledWith({ x: 40, y: 60 });
    expect(h.openTextEdit).not.toHaveBeenCalled();
  });

  it('opens an edit-mode session when the click hits an existing text layer', () => {
    const doc = makeDoc({ layers: [textLayer()] });
    const h = createHarness(doc);
    const tool = createTextTool();

    // 'hello' at 20px → estimated block ~60×24, so (5,5) is inside.
    down(tool, h.ctx, pointer(5, 5));

    expect(h.openTextEdit).toHaveBeenCalledTimes(1);
    expect(h.openTextEdit).toHaveBeenCalledWith('text-existing');
    expect(h.openTextCreate).not.toHaveBeenCalled();
  });

  it('creates rather than edits when the click misses the text block', () => {
    const doc = makeDoc({ layers: [textLayer()] });
    const h = createHarness(doc);
    const tool = createTextTool();

    down(tool, h.ctx, pointer(500, 500));

    expect(h.openTextEdit).not.toHaveBeenCalled();
    expect(h.openTextCreate).toHaveBeenCalledTimes(1);
  });

  it('does not edit a locked text layer (falls through to create)', () => {
    const doc = makeDoc({ layers: [textLayer({ isLocked: true })] });
    const h = createHarness(doc);
    const tool = createTextTool();

    down(tool, h.ctx, pointer(5, 5));

    expect(h.openTextEdit).not.toHaveBeenCalled();
    expect(h.openTextCreate).toHaveBeenCalledTimes(1);
  });

  it('does not edit a hidden (disabled) text layer', () => {
    const doc = makeDoc({ layers: [textLayer({ isEnabled: false })] });
    const h = createHarness(doc);
    const tool = createTextTool();

    down(tool, h.ctx, pointer(5, 5));

    expect(h.openTextEdit).not.toHaveBeenCalled();
    expect(h.openTextCreate).toHaveBeenCalledTimes(1);
  });

  it('is a no-op while a session is already open (the click blurs/commits React-side)', () => {
    const h = createHarness(makeDoc());
    h.stores.textEditSession.set({
      id: 1,
      layerId: null,
      mode: 'create',
      source: {
        align: 'left',
        color: '#000000',
        content: '',
        fontFamily: 'Inter',
        fontSize: 20,
        fontWeight: 400,
        lineHeight: 1.2,
        type: 'text',
      },
      startSource: null,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    });
    const tool = createTextTool();

    down(tool, h.ctx, pointer(40, 60));

    expect(h.openTextCreate).not.toHaveBeenCalled();
    expect(h.openTextEdit).not.toHaveBeenCalled();
  });

  it('ignores a non-primary button press', () => {
    const h = createHarness(makeDoc());
    const tool = createTextTool();

    down(tool, h.ctx, pointer(40, 60, 2));

    expect(h.openTextCreate).not.toHaveBeenCalled();
  });
});

describe('text tool: teardown', () => {
  it('cancels the session on a real deactivate but not on a temporary one', () => {
    const h = createHarness(makeDoc());
    const tool = createTextTool();

    tool.onDeactivate?.(h.ctx, { temporary: true });
    expect(h.cancelTextEdit).not.toHaveBeenCalled();

    tool.onDeactivate?.(h.ctx);
    expect(h.cancelTextEdit).toHaveBeenCalledTimes(1);
  });

  it('cancels the session on an Escape key command', () => {
    const h = createHarness(makeDoc());
    const tool = createTextTool();

    tool.onKeyCommand?.(h.ctx, 'cancel');
    expect(h.cancelTextEdit).toHaveBeenCalledTimes(1);

    // Apply is a no-op here (React owns the live content/commit).
    tool.onKeyCommand?.(h.ctx, 'apply');
    expect(h.cancelTextEdit).toHaveBeenCalledTimes(1);
  });
});
