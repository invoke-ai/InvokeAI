import {
  DEFAULT_BRUSH_OPTIONS,
  DEFAULT_ERASER_OPTIONS,
  DEFAULT_SHAPE_OPTIONS,
  DEFAULT_TEXT_OPTIONS,
  MAX_BRUSH_SIZE,
} from '@workbench/canvas-engine/api';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  bindCanvasToolMemory,
  type CanvasToolMemory,
  normalizeCanvasToolMemory,
  TOOL_MEMORY_WRITE_DELAY_MS,
} from './toolMemory';

const { color: _color, ...DEFAULT_BRUSH } = DEFAULT_BRUSH_OPTIONS;

afterEach(() => {
  vi.useRealTimers();
});

describe('normalizeCanvasToolMemory', () => {
  it('reads nothing stored as the engine defaults', () => {
    for (const raw of [undefined, null, 'brush', [], {}]) {
      expect(normalizeCanvasToolMemory(raw)).toEqual({
        brush: DEFAULT_BRUSH,
        eraser: DEFAULT_ERASER_OPTIONS,
        shape: DEFAULT_SHAPE_OPTIONS,
        text: DEFAULT_TEXT_OPTIONS,
      });
    }
  });

  it('keeps valid fields, clamps ranges and replaces the rest field by field', () => {
    const memory = normalizeCanvasToolMemory({
      brush: { hardness: 0.25, opacity: 7, pressureAffectsWidth: false, size: 99_999 },
      eraser: { size: 'large' },
      shape: { kind: 'hexagon', strokeEnabled: true, strokeWidth: 12, target: 'new' },
      text: {
        align: 'center',
        fontFamily: '',
        fontRef: { contentHash: 'h', family: 'Fam', id: 'font-1', label: 'Fam' },
        fontSize: 72,
        fontVariations: { slnt: -5, wght: 'bold' },
        fontWeight: 650.4,
      },
    });
    expect(memory.brush).toEqual({
      ...DEFAULT_BRUSH,
      hardness: 0.25,
      opacity: 1,
      pressureAffectsWidth: false,
      size: MAX_BRUSH_SIZE,
    });
    expect(memory.eraser).toEqual(DEFAULT_ERASER_OPTIONS);
    expect(memory.shape).toEqual({ ...DEFAULT_SHAPE_OPTIONS, strokeEnabled: true, strokeWidth: 12, target: 'new' });
    expect(memory.text).toEqual({
      ...DEFAULT_TEXT_OPTIONS,
      align: 'center',
      fontRef: { contentHash: 'h', family: 'Fam', id: 'font-1', label: 'Fam' },
      fontSize: 72,
      fontVariations: { slnt: -5 },
      fontWeight: 650,
    });
  });
});

describe('bindCanvasToolMemory', () => {
  const stored: CanvasToolMemory = {
    brush: { ...DEFAULT_BRUSH, size: 12 },
    eraser: { ...DEFAULT_ERASER_OPTIONS, hardness: 0.5 },
    shape: { ...DEFAULT_SHAPE_OPTIONS, kind: 'star' },
    text: { ...DEFAULT_TEXT_OPTIONS, fontSize: 20 },
  };

  it('starts a new engine with the stored options, keeping the brush color the color pair gave it', () => {
    const stores = createEngineStores();
    stores.brushOptions.set({ ...stores.brushOptions.get(), color: '#ff0000' });
    bindCanvasToolMemory(stores, { read: () => stored, write: vi.fn() });

    expect(stores.brushOptions.get()).toEqual({ ...stored.brush, color: '#ff0000' });
    expect(stores.eraserOptions.get()).toEqual(stored.eraser);
    expect(stores.shapeOptions.get().kind).toBe('star');
    expect(stores.textOptions.get().fontSize).toBe(20);
  });

  it('saves the changed tools once quiet, ignores a brush color change, and saves a pending change on disposal', () => {
    vi.useFakeTimers();
    const stores = createEngineStores();
    const write = vi.fn();
    const binding = bindCanvasToolMemory(stores, { read: () => stored, write });

    stores.brushOptions.set({ ...stores.brushOptions.get(), size: 30 });
    stores.brushOptions.set({ ...stores.brushOptions.get(), size: 40 });
    vi.advanceTimersByTime(TOOL_MEMORY_WRITE_DELAY_MS - 1);
    expect(write).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(write).toHaveBeenCalledExactlyOnceWith({ brush: { ...stored.brush, size: 40 } });

    stores.brushOptions.set({ ...stores.brushOptions.get(), color: '#00ff00' });
    vi.advanceTimersByTime(TOOL_MEMORY_WRITE_DELAY_MS);
    expect(write).toHaveBeenCalledOnce();

    stores.eraserOptions.set({ ...stores.eraserOptions.get(), size: 5 });
    binding.dispose();
    expect(write).toHaveBeenLastCalledWith({ eraser: { ...stored.eraser, size: 5 } });
    stores.shapeOptions.set({ ...stores.shapeOptions.get(), kind: 'rect' });
    vi.advanceTimersByTime(TOOL_MEMORY_WRITE_DELAY_MS);
    expect(write).toHaveBeenCalledTimes(2);
  });

  it('takes in what another canvas stored, except a change of its own still waiting to be saved', () => {
    vi.useFakeTimers();
    let memory: Record<string, unknown> = { ...stored };
    const listeners = new Set<() => void>();
    const stores = createEngineStores();
    const write = vi.fn();
    bindCanvasToolMemory(stores, {
      read: () => memory,
      subscribe: (listener) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      write,
    });

    stores.shapeOptions.set({ ...stores.shapeOptions.get(), strokeWidth: 9 });
    memory = { ...memory, brush: { ...stored.brush, size: 77 }, shape: { ...stored.shape, kind: 'ellipse' } };
    listeners.forEach((listener) => listener());

    expect(stores.brushOptions.get().size).toBe(77);
    expect(stores.shapeOptions.get()).toMatchObject({ kind: 'star', strokeWidth: 9 });
    vi.advanceTimersByTime(TOOL_MEMORY_WRITE_DELAY_MS);
    // Taking in the stored brush is not a change of this canvas's to save back.
    expect(write).toHaveBeenCalledExactlyOnceWith({ shape: { ...stored.shape, strokeWidth: 9 } });
  });
});
