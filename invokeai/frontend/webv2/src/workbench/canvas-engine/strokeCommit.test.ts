import type { StrokeCommittedEvent } from '@workbench/canvas-engine/tools/tool';

import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { createStrokeCommit, type CreateStrokeCommitDeps } from './strokeCommit';

/** Dimensions must match the event's dirty rect — `createImagePatchEntry` asserts it. */
const imageData = (bytes = 4): ImageData =>
  ({ data: { byteLength: bytes }, height: 4, width: 4 }) as unknown as ImageData;

const strokeEvent = (overrides: Partial<StrokeCommittedEvent> = {}): StrokeCommittedEvent =>
  ({
    afterImageData: imageData(),
    beforeImageData: imageData(),
    createdLayer: undefined,
    dirtyRect: { height: 4, width: 4, x: 1, y: 2 },
    layerId: 'layer-1',
    tool: 'brush',
    ...overrides,
  }) as unknown as StrokeCommittedEvent;

type TestDeps = {
  -readonly [K in keyof CreateStrokeCommitDeps]: CreateStrokeCommitDeps[K];
} & { history: { isApplying: ReturnType<typeof vi.fn>; push: ReturnType<typeof vi.fn> } };

let deps: TestDeps;
let listener: Mock<(event: StrokeCommittedEvent) => void>;

beforeEach(() => {
  listener = vi.fn<(event: StrokeCommittedEvent) => void>();
  deps = {
    applyImagePatch: vi.fn(),
    commitPaintEdit: vi.fn(),
    dispatchCanvasMutation: vi.fn(() => true),
    endNudgeBurst: vi.fn(),
    history: { isApplying: vi.fn(() => false), push: vi.fn() },
    layerCache: { getOrCreateRect: vi.fn(() => ({ stale: true })) } as unknown as CreateStrokeCommitDeps['layerCache'],
    markLayerDirty: vi.fn(),
    notifyLayerPainted: vi.fn(),
    strokeListeners: new Set<(event: StrokeCommittedEvent) => void>([listener]),
  } as unknown as TestDeps;
});

describe('commitOrdinaryStroke', () => {
  it('ends the nudge burst, repaints, and marks the layer dirty before recording history', () => {
    createStrokeCommit(deps).commitOrdinaryStroke(strokeEvent());
    expect(deps.endNudgeBurst).toHaveBeenCalled();
    expect(deps.notifyLayerPainted).toHaveBeenCalledWith('layer-1');
    expect(deps.markLayerDirty).toHaveBeenCalledWith('layer-1');
    expect(deps.history.push).toHaveBeenCalledTimes(1);
  });

  it('labels the entry by the committing tool', () => {
    createStrokeCommit(deps).commitOrdinaryStroke(strokeEvent({ tool: 'eraser' } as Partial<StrokeCommittedEvent>));
    expect(deps.history.push.mock.calls[0]?.[0]).toMatchObject({ label: 'Eraser stroke' });

    deps.history.push.mockClear();
    createStrokeCommit(deps).commitOrdinaryStroke(strokeEvent());
    expect(deps.history.push.mock.calls[0]?.[0]).toMatchObject({ label: 'Brush stroke' });
  });

  it('records no history while a replay is applying, but still repaints and notifies', () => {
    deps.history.isApplying = vi.fn(() => true);
    createStrokeCommit(deps).commitOrdinaryStroke(strokeEvent());
    expect(deps.history.push).not.toHaveBeenCalled();
    expect(deps.commitPaintEdit).not.toHaveBeenCalled();
    expect(deps.notifyLayerPainted).toHaveBeenCalled();
    expect(listener).toHaveBeenCalled();
  });

  it('fans the event out to every subscriber', () => {
    const second = vi.fn<(event: StrokeCommittedEvent) => void>();
    deps.strokeListeners = new Set<(event: StrokeCommittedEvent) => void>([listener, second]);
    const event = strokeEvent();
    createStrokeCommit(deps).commitOrdinaryStroke(event);
    expect(listener).toHaveBeenCalledWith(event);
    expect(second).toHaveBeenCalledWith(event);
  });
});

describe('a stroke that auto-created its layer', () => {
  const created = { index: 2, layer: { id: 'layer-1' } as never };

  it('undoes by removing the layer, not just the pixels', () => {
    createStrokeCommit(deps).commitOrdinaryStroke(strokeEvent({ createdLayer: created } as never));
    const entry = deps.history.push.mock.calls[0]?.[0] as { undo(): void };
    entry.undo();
    expect(deps.dispatchCanvasMutation).toHaveBeenCalledWith({ ids: ['layer-1'], type: 'removeCanvasLayers' });
  });

  it('redoes by re-adding the layer at its index, then writing the pixels back', () => {
    createStrokeCommit(deps).commitOrdinaryStroke(strokeEvent({ createdLayer: created } as never));
    const entry = deps.history.push.mock.calls[0]?.[0] as { redo(): void };
    entry.redo();
    expect(deps.dispatchCanvasMutation).toHaveBeenCalledWith({
      index: 2,
      layer: created.layer,
      type: 'addCanvasLayer',
    });
    expect(deps.applyImagePatch).toHaveBeenCalledWith(
      'layer-1',
      { height: 4, width: 4, x: 1, y: 2 },
      expect.anything()
    );
  });

  it('marks the re-created cache fresh so an async rasterize cannot clobber the restored stroke', () => {
    const entry_ = { stale: true };
    deps.layerCache = { getOrCreateRect: vi.fn(() => entry_) } as unknown as CreateStrokeCommitDeps['layerCache'];
    createStrokeCommit(deps).commitOrdinaryStroke(strokeEvent({ createdLayer: created } as never));
    const entry = deps.history.push.mock.calls[0]?.[0] as { redo(): void };
    entry.redo();
    expect(entry_.stale).toBe(false);
  });

  it('sizes the entry from both pixel buffers plus overhead', () => {
    createStrokeCommit(deps).commitOrdinaryStroke(
      strokeEvent({ afterImageData: imageData(10), beforeImageData: imageData(6), createdLayer: created } as never)
    );
    expect(deps.history.push.mock.calls[0]?.[0]).toMatchObject({ bytes: 6 + 10 + 256 });
  });
});
