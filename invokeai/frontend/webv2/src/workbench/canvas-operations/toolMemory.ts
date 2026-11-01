/**
 * Tool options a person tunes once and expects to find again: brush, eraser, shape and text settings. They belong to
 * the account (its preferences), not to one engine, so a canvas whose engine was disposed after its project went out
 * of view, or another project's canvas, starts with them. Brush color is not here: it follows the project's color
 * pair. Stored values are normalized against the engine defaults on every read, so an older or hand-edited record
 * never reaches a store.
 */

import type {
  BrushOptions,
  CanvasTextFontRef,
  CanvasTextFontStyle,
  EraserOptions,
  ShapeToolKind,
  ShapeToolOptions,
  TextToolOptions,
} from '@workbench/canvas-engine/api';

import {
  DEFAULT_BRUSH_OPTIONS,
  DEFAULT_ERASER_OPTIONS,
  DEFAULT_SHAPE_OPTIONS,
  DEFAULT_TEXT_OPTIONS,
  MAX_BRUSH_SIZE,
  MAX_SHAPE_STROKE_WIDTH,
  MAX_TEXT_FONT_SIZE,
  MIN_BRUSH_SIZE,
} from '@workbench/canvas-engine/api';

/** A type, not an interface, so it reads as the plain record the preferences store keeps. */
export type CanvasToolMemory = {
  brush: Omit<BrushOptions, 'color'>;
  eraser: EraserOptions;
  shape: ShapeToolOptions;
  text: TextToolOptions;
};

/** Where the account keeps the memory; the composition root injects the preferences store. */
export interface CanvasToolMemoryPort {
  read(): unknown;
  /** Stores the tools that changed; the others keep what is stored, which another canvas may have saved since. */
  write(changed: Partial<CanvasToolMemory>): void;
  /** Calls back when the stored memory may have changed: another canvas saved, or the account's settings loaded. */
  subscribe?(listener: () => void): () => void;
}

/** Quiet time after the last change before the memory is written, so a size scrub saves once. */
export const TOOL_MEMORY_WRITE_DELAY_MS = 500;

const SHAPE_KINDS: readonly ShapeToolKind[] = ['rect', 'ellipse', 'triangle', 'star', 'polygon', 'freehand'];
const FONT_STYLES: readonly CanvasTextFontStyle[] = ['normal', 'italic', 'oblique'];
const TEXT_ALIGNS: readonly TextToolOptions['align'][] = ['left', 'center', 'right'];
const FONT_WEIGHTS = { max: 1000, min: 1 };
const LINE_HEIGHTS = { max: 10, min: 0.5 };
const MAX_FONT_FAMILY_LENGTH = 512;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const number = (value: unknown, fallback: number, min: number, max: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback;

const boolean = (value: unknown, fallback: boolean): boolean => (typeof value === 'boolean' ? value : fallback);

const oneOf = <T extends string>(value: unknown, options: readonly T[], fallback: T): T =>
  options.includes(value as T) ? (value as T) : fallback;

const fontRef = (value: unknown): CanvasTextFontRef | undefined =>
  isRecord(value) &&
  typeof value.id === 'string' &&
  typeof value.contentHash === 'string' &&
  typeof value.family === 'string' &&
  typeof value.label === 'string'
    ? { contentHash: value.contentHash, family: value.family, id: value.id, label: value.label }
    : undefined;

const fontVariations = (value: unknown): Record<string, number> =>
  isRecord(value)
    ? Object.fromEntries(
        Object.entries(value).filter((entry): entry is [string, number] => Number.isFinite(entry[1] as number))
      )
    : {};

/** The stored memory with every missing, mistyped or out-of-range field replaced by its default. */
export const normalizeCanvasToolMemory = (raw: unknown): CanvasToolMemory => {
  const stored = isRecord(raw) ? raw : {};
  const brush = isRecord(stored.brush) ? stored.brush : {};
  const eraser = isRecord(stored.eraser) ? stored.eraser : {};
  const shape = isRecord(stored.shape) ? stored.shape : {};
  const text = isRecord(stored.text) ? stored.text : {};
  const ref = fontRef(text.fontRef);
  return {
    brush: {
      hardness: number(brush.hardness, DEFAULT_BRUSH_OPTIONS.hardness, 0, 1),
      opacity: number(brush.opacity, DEFAULT_BRUSH_OPTIONS.opacity, 0, 1),
      pressureAffectsOpacity: boolean(brush.pressureAffectsOpacity, DEFAULT_BRUSH_OPTIONS.pressureAffectsOpacity),
      pressureAffectsWidth: boolean(brush.pressureAffectsWidth, DEFAULT_BRUSH_OPTIONS.pressureAffectsWidth),
      size: number(brush.size, DEFAULT_BRUSH_OPTIONS.size, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE),
    },
    eraser: {
      hardness: number(eraser.hardness, DEFAULT_ERASER_OPTIONS.hardness, 0, 1),
      opacity: number(eraser.opacity, DEFAULT_ERASER_OPTIONS.opacity, 0, 1),
      pressureAffectsWidth: boolean(eraser.pressureAffectsWidth, DEFAULT_ERASER_OPTIONS.pressureAffectsWidth),
      size: number(eraser.size, DEFAULT_ERASER_OPTIONS.size, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE),
    },
    shape: {
      fillEnabled: boolean(shape.fillEnabled, DEFAULT_SHAPE_OPTIONS.fillEnabled),
      kind: oneOf(shape.kind, SHAPE_KINDS, DEFAULT_SHAPE_OPTIONS.kind),
      strokeEnabled: boolean(shape.strokeEnabled, DEFAULT_SHAPE_OPTIONS.strokeEnabled),
      strokeWidth: number(shape.strokeWidth, DEFAULT_SHAPE_OPTIONS.strokeWidth, 1, MAX_SHAPE_STROKE_WIDTH),
      target: oneOf(shape.target, ['selected', 'new'] as const, DEFAULT_SHAPE_OPTIONS.target),
    },
    text: {
      align: oneOf(text.align, TEXT_ALIGNS, DEFAULT_TEXT_OPTIONS.align),
      fontFamily:
        typeof text.fontFamily === 'string' &&
        text.fontFamily.trim() &&
        text.fontFamily.length <= MAX_FONT_FAMILY_LENGTH
          ? text.fontFamily
          : DEFAULT_TEXT_OPTIONS.fontFamily,
      fontSize: number(text.fontSize, DEFAULT_TEXT_OPTIONS.fontSize, 1, MAX_TEXT_FONT_SIZE),
      fontStyle: oneOf(text.fontStyle, FONT_STYLES, DEFAULT_TEXT_OPTIONS.fontStyle ?? 'normal'),
      fontVariations: fontVariations(text.fontVariations),
      fontWeight: Math.round(
        number(text.fontWeight, DEFAULT_TEXT_OPTIONS.fontWeight, FONT_WEIGHTS.min, FONT_WEIGHTS.max)
      ),
      lineHeight: number(text.lineHeight, DEFAULT_TEXT_OPTIONS.lineHeight, LINE_HEIGHTS.min, LINE_HEIGHTS.max),
      ...(ref ? { fontRef: ref } : {}),
    },
  };
};

interface OptionStore<T> {
  get(): T;
  set(next: T): void;
  subscribe(listener: () => void): () => void;
}

/** The engine's option stores this memory reads and seeds. */
export interface ToolMemoryStores {
  brushOptions: OptionStore<BrushOptions>;
  eraserOptions: OptionStore<EraserOptions>;
  shapeOptions: OptionStore<ShapeToolOptions>;
  textOptions: OptionStore<TextToolOptions>;
}

type ToolKey = keyof CanvasToolMemory;

const TOOL_KEYS: readonly ToolKey[] = ['brush', 'eraser', 'shape', 'text'];

const capture = <Key extends ToolKey>(stores: ToolMemoryStores, key: Key): CanvasToolMemory[Key] => {
  if (key === 'brush') {
    const { color: _color, ...brush } = stores.brushOptions.get();
    return brush as CanvasToolMemory[Key];
  }
  return stores[`${key}Options` as 'eraserOptions' | 'shapeOptions' | 'textOptions'].get() as CanvasToolMemory[Key];
};

const sameValue = (a: unknown, b: unknown): boolean => {
  if (Object.is(a, b)) {
    return true;
  }
  if (!isRecord(a) || !isRecord(b)) {
    return false;
  }
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every((key) => sameValue(a[key], b[key]));
};

export interface CanvasToolMemoryBinding {
  /** Writes a change still waiting out its delay; called on disposal. Idempotent. */
  dispose(): void;
}

/**
 * Seeds an engine's option stores from the account's memory and keeps them in step with it, then writes each tool
 * that changed back once the stores have been quiet for {@link TOOL_MEMORY_WRITE_DELAY_MS}. Writing only the changed
 * tools, and taking in what another canvas stored for the rest, keeps two engines (a warm one waiting out its grace
 * period and the one now open) from overwriting each other with stale options. Disposal writes a pending change at
 * once and stops listening.
 */
export const bindCanvasToolMemory = (
  stores: ToolMemoryStores,
  port: CanvasToolMemoryPort,
  timers: Pick<typeof globalThis, 'clearTimeout' | 'setTimeout'> = globalThis
): CanvasToolMemoryBinding => {
  /** Tools changed here and not yet written: the stored memory does not overwrite them meanwhile. */
  const dirty = new Set<ToolKey>();
  let seeding = false;
  let lastRead: unknown;
  const seed = (): void => {
    const raw = port.read();
    if (raw === lastRead) {
      return;
    }
    lastRead = raw;
    const memory = normalizeCanvasToolMemory(raw);
    seeding = true;
    try {
      for (const key of TOOL_KEYS) {
        if (dirty.has(key) || sameValue(capture(stores, key), memory[key])) {
          continue;
        }
        if (key === 'brush') {
          stores.brushOptions.set({ ...memory.brush, color: stores.brushOptions.get().color });
        } else {
          stores[`${key}Options` as 'eraserOptions'].set(memory[key] as EraserOptions);
        }
      }
    } finally {
      seeding = false;
    }
  };
  seed();

  let pending: ReturnType<typeof setTimeout> | null = null;
  const write = (): void => {
    pending = null;
    const changed: Partial<CanvasToolMemory> = {};
    for (const key of dirty) {
      Object.assign(changed, { [key]: capture(stores, key) });
    }
    dirty.clear();
    port.write(changed);
  };
  // A brush color change alone is the color pair's, which the project keeps.
  let lastBrush = capture(stores, 'brush');
  const onChange = (key: ToolKey) => (): void => {
    if (key === 'brush') {
      const brush = capture(stores, 'brush');
      if (sameValue(brush, lastBrush)) {
        return;
      }
      lastBrush = brush;
    }
    if (seeding) {
      return;
    }
    dirty.add(key);
    if (pending !== null) {
      timers.clearTimeout(pending);
    }
    pending = timers.setTimeout(write, TOOL_MEMORY_WRITE_DELAY_MS);
  };
  const unsubscribes = [
    stores.brushOptions.subscribe(onChange('brush')),
    stores.eraserOptions.subscribe(onChange('eraser')),
    stores.shapeOptions.subscribe(onChange('shape')),
    stores.textOptions.subscribe(onChange('text')),
    port.subscribe?.(seed) ?? (() => undefined),
  ];
  let disposed = false;
  return {
    dispose: () => {
      if (disposed) {
        return;
      }
      disposed = true;
      unsubscribes.forEach((unsubscribe) => unsubscribe());
      if (pending !== null) {
        timers.clearTimeout(pending);
        write();
      }
    },
  };
};
