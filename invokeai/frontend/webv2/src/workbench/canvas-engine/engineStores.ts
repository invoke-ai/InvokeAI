/**
 * Per-engine transient external stores.
 *
 * These are the narrow, imperative channels React subscribes to (in the widget
 * task) to observe engine-owned interaction state — active tool, zoom,
 * readiness, cursor, and per-layer thumbnail versions — without the engine ever
 * importing React. They follow the `externalStore.ts` pattern (a listener
 * channel plus a snapshot getter, `useSyncExternalStore`-compatible) but
 * deliberately do NOT import it: `externalStore.ts` pulls in React, and this
 * module must stay node-safe and React-free. The React hooks live with the
 * widget shell.
 *
 * Zero React, zero import-time side effects.
 */

import type { SamInteractionState } from '@workbench/canvas-engine/samInteraction';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect, SelectionOp, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasLayerSourceContract } from '@workbench/types';

import { type CheckerColors, DEFAULT_CHECKER_COLORS } from '@workbench/canvas-engine/render/compositor';

export type { CheckerColors };

/** A `text` layer source (content + style params). */
export type TextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;

/** Brush tool options (document-space size, style, and pressure behavior). */
export interface BrushOptions {
  /** Base stroke diameter in document units. */
  size: number;
  /** Fill color (any CSS color string). */
  color: string;
  /** Per-stroke opacity in [0, 1]. */
  opacity: number;
  /** Whether pen pressure modulates the stroke width. */
  pressureSensitivity: boolean;
}

/** Eraser tool options. */
export interface EraserOptions {
  /** Base eraser diameter in document units. */
  size: number;
  /** Per-stroke erase strength in [0, 1]. */
  opacity: number;
}

/** Lasso (selection) tool options: the boolean op applied when a path commits. */
export interface LassoToolOptions {
  /** The op a committed lasso path applies to the selection, when no modifier overrides it. */
  mode: SelectionOp;
}

/** Default lasso options: a fresh path replaces the selection. */
export const DEFAULT_LASSO_OPTIONS: LassoToolOptions = {
  mode: 'replace',
};

/** A single gradient stop (offset in [0,1], any CSS color). */
export interface GradientStop {
  offset: number;
  color: string;
}

/**
 * Shape tool options: the kind drawn on the next drag, and the fill/stroke
 * style applied to it (and, when a shape layer is selected, edited live on it).
 * `fill`/`stroke` are `null` for "none".
 */
export interface ShapeToolOptions {
  kind: 'rect' | 'ellipse';
  fill: string | null;
  stroke: string | null;
  strokeWidth: number;
}

/** Sensible starting shape options: a filled black rect, no stroke. */
export const DEFAULT_SHAPE_OPTIONS: ShapeToolOptions = {
  fill: '#000000',
  kind: 'rect',
  stroke: null,
  strokeWidth: 8,
};

/** Largest shape stroke width (document px) the options bar clamps to. */
export const MAX_SHAPE_STROKE_WIDTH = 2000;

/**
 * Gradient tool options: the kind, the linear angle (degrees), and the stops.
 * The minimal two-stop editor edits `stops[0]` (start) and the last stop (end);
 * a full multi-stop editor is a follow-up.
 */
export interface GradientToolOptions {
  kind: 'linear' | 'radial';
  angle: number;
  stops: GradientStop[];
}

/** Sensible starting gradient options: black→transparent, horizontal linear. */
export const DEFAULT_GRADIENT_OPTIONS: GradientToolOptions = {
  angle: 0,
  kind: 'linear',
  stops: [
    { color: '#000000ff', offset: 0 },
    { color: '#00000000', offset: 1 },
  ],
};

/** The text style the text tool applies to a newly created layer (and edits live). */
export interface TextToolOptions {
  fontFamily: string;
  fontSize: number;
  /** CSS numeric weight (400/500/600/700). */
  fontWeight: number;
  /** Unitless line-height multiplier over `fontSize`. */
  lineHeight: number;
  align: 'left' | 'center' | 'right';
  color: string;
}

/**
 * A small curated font list offered by the text options bar. Values are CSS
 * `font-family` stacks so each falls back gracefully; keep it short this phase
 * (a full system-font enumeration is a follow-up).
 */
export const TEXT_FONT_FAMILIES: readonly { label: string; value: string }[] = [
  { label: 'Inter', value: "'Inter Variable', Inter, sans-serif" },
  { label: 'Sans-serif', value: 'system-ui, sans-serif' },
  { label: 'Serif', value: "Georgia, 'Times New Roman', serif" },
  { label: 'Monospace', value: "'JetBrains Mono', ui-monospace, monospace" },
];

/** Weights the text options bar offers. */
export const TEXT_FONT_WEIGHTS: readonly number[] = [400, 500, 600, 700];

/** Smallest / largest font size (document px) the text options bar clamps to. */
export const MIN_TEXT_FONT_SIZE = 1;
export const MAX_TEXT_FONT_SIZE = 2000;

/** Sensible starting text options: black left-aligned Inter at 48px. */
export const DEFAULT_TEXT_OPTIONS: TextToolOptions = {
  align: 'left',
  color: '#000000',
  fontFamily: TEXT_FONT_FAMILIES[0]!.value,
  fontSize: 48,
  fontWeight: 400,
  lineHeight: 1.2,
};

/** Bbox (generation-frame) tool options: the aspect-ratio lock. */
export interface BboxToolOptions {
  /** Whether corner/edge resize preserves {@link BboxToolOptions.aspectRatio}. */
  aspectLocked: boolean;
  /** The locked width / height ratio. */
  aspectRatio: number;
}

/** Default grid size (document px) the bbox snaps to before a model feeds a real one. */
export const DEFAULT_BBOX_GRID = 8;

/** Default bbox tool options: aspect unlocked, square ratio. */
export const DEFAULT_BBOX_OPTIONS: BboxToolOptions = {
  aspectLocked: false,
  aspectRatio: 1,
};

/** Smallest and largest brush/eraser diameters (document units) the size step clamps to. */
export const MIN_BRUSH_SIZE = 1;
export const MAX_BRUSH_SIZE = 2000;

/** Sensible starting brush options. */
export const DEFAULT_BRUSH_OPTIONS: BrushOptions = {
  color: '#000000',
  opacity: 1,
  pressureSensitivity: true,
  size: 50,
};

/** Sensible starting eraser options. */
export const DEFAULT_ERASER_OPTIONS: EraserOptions = {
  opacity: 1,
  size: 50,
};

/**
 * An active transform-tool session on one layer. Outlives individual pointer
 * gestures (drag handles, adjust numerics) until Apply or Cancel. `startTransform`
 * is the committed transform captured at session start (restored on Cancel /
 * used as the undo inverse); `transform` is the live, edited transform the
 * compositor previews and the options bar renders as numerics.
 */
export interface TransformSession {
  layerId: string;
  startTransform: LayerTransform;
  transform: LayerTransform;
}

/**
 * An active text-editing session. Set by the text tool; while it is active the
 * contenteditable portal (in `widgets/canvas`) shows the live text and the
 * compositor SKIPS the session's layer (edit mode) so the two don't double-draw.
 *
 * Two modes:
 * - **create**: no layer exists yet (`layerId === null`, `startSource === null`).
 *   Commit dispatches ONE `addCanvasLayer` with the final content; cancel adds
 *   nothing. This keeps a new text layer to a single, cleanly-undoable commit.
 * - **edit**: an existing text layer is being re-edited. `startSource` is its
 *   committed source (the exact undo inverse / no-change baseline); `source` is
 *   the live, style-edited source. Commit dispatches ONE `updateCanvasLayerSource`.
 *
 * `source` carries the live style (font/size/weight/lineHeight/align/color) the
 * portal renders WYSIWYG and the options bar edits; `content` on it is only the
 * seed — the live typed content lives in the contenteditable DOM until commit.
 * `transform` positions/scales the portal (document→screen via the view matrix).
 * `id` increments per session so React can key (remount) the editable per open.
 */
export interface TextEditSession {
  id: number;
  mode: 'create' | 'edit';
  layerId: string | null;
  startSource: TextSource | null;
  source: TextSource;
  transform: LayerTransform;
}

/** A single-value store, `useSyncExternalStore`-compatible. */
export interface ScalarStore<T> {
  get(): T;
  set(next: T): void;
  subscribe(listener: () => void): () => void;
}

const createScalarStore = <T>(initial: T, isEqual: (a: T, b: T) => boolean = Object.is): ScalarStore<T> => {
  let value = initial;
  const listeners = new Set<() => void>();

  return {
    get: () => value,
    set: (next) => {
      if (isEqual(value, next)) {
        return;
      }
      value = next;
      for (const listener of listeners) {
        listener();
      }
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
};

/**
 * A keyed numeric store with per-key subscription granularity, so a React
 * component watching one layer's thumbnail version only re-renders when that
 * layer changes. A global `subscribe` is also exposed for coarse observers.
 */
export interface KeyedVersionStore {
  get(key: string): number | undefined;
  set(key: string, value: number): void;
  delete(key: string): void;
  /** Subscribes to changes for a single key. */
  subscribeKey(key: string, listener: () => void): () => void;
  /** Subscribes to any change across all keys. */
  subscribe(listener: () => void): () => void;
}

export type LayerThumbnailStatus = 'loading' | 'ready' | 'error';

/** A per-layer thumbnail request state; absence represents `idle`. */
export interface KeyedThumbnailStatusStore {
  get(key: string): LayerThumbnailStatus | undefined;
  set(key: string, value: LayerThumbnailStatus): void;
  delete(key: string): void;
  clear(): void;
  subscribeKey(key: string, listener: () => void): () => void;
}

const createKeyedVersionStore = (): KeyedVersionStore => {
  const values = new Map<string, number>();
  const keyedListeners = new Map<string, Set<() => void>>();
  const globalListeners = new Set<() => void>();

  const notify = (key: string): void => {
    for (const listener of keyedListeners.get(key) ?? []) {
      listener();
    }
    for (const listener of globalListeners) {
      listener();
    }
  };

  return {
    delete: (key) => {
      if (values.delete(key)) {
        notify(key);
      }
    },
    get: (key) => values.get(key),
    set: (key, value) => {
      if (values.get(key) === value) {
        return;
      }
      values.set(key, value);
      notify(key);
    },
    subscribe: (listener) => {
      globalListeners.add(listener);
      return () => {
        globalListeners.delete(listener);
      };
    },
    subscribeKey: (key, listener) => {
      const listeners = keyedListeners.get(key) ?? new Set<() => void>();
      listeners.add(listener);
      keyedListeners.set(key, listeners);
      return () => {
        listeners.delete(listener);
        if (listeners.size === 0) {
          keyedListeners.delete(key);
        }
      };
    },
  };
};

const createKeyedThumbnailStatusStore = (): KeyedThumbnailStatusStore => {
  const values = new Map<string, LayerThumbnailStatus>();
  const keyedListeners = new Map<string, Set<() => void>>();
  const notify = (key: string): void => {
    for (const listener of keyedListeners.get(key) ?? []) {
      listener();
    }
  };

  return {
    clear: () => {
      const keys = [...values.keys()];
      values.clear();
      for (const key of keys) {
        notify(key);
      }
    },
    delete: (key) => {
      if (values.delete(key)) {
        notify(key);
      }
    },
    get: (key) => values.get(key),
    set: (key, value) => {
      if (values.get(key) === value) {
        return;
      }
      values.set(key, value);
      notify(key);
    },
    subscribeKey: (key, listener) => {
      const listeners = keyedListeners.get(key) ?? new Set<() => void>();
      listeners.add(listener);
      keyedListeners.set(key, listeners);
      return () => {
        listeners.delete(listener);
        if (listeners.size === 0) {
          keyedListeners.delete(key);
        }
      };
    },
  };
};

/** The bundle of transient stores owned by one engine instance. */
export interface EngineStores {
  activeTool: ScalarStore<ToolId>;
  zoom: ScalarStore<number>;
  viewportReady: ScalarStore<boolean>;
  cursor: ScalarStore<string>;
  thumbnailVersion: KeyedVersionStore;
  thumbnailStatus: KeyedThumbnailStatusStore;
  /** Brush tool options (size / color / opacity / pressure). */
  brushOptions: ScalarStore<BrushOptions>;
  /** Eraser tool options (size / opacity). */
  eraserOptions: ScalarStore<EraserOptions>;
  /** Whether the engine-owned canvas history has an entry to undo. */
  canUndo: ScalarStore<boolean>;
  /** Whether the engine-owned canvas history has an entry to redo. */
  canRedo: ScalarStore<boolean>;
  /** Bbox tool options (aspect lock / ratio). */
  bboxOptions: ScalarStore<BboxToolOptions>;
  /** Lasso tool options (the committed boolean op mode). */
  lassoOptions: ScalarStore<LassoToolOptions>;
  /** Shape tool options (kind / fill / stroke / stroke width). */
  shapeOptions: ScalarStore<ShapeToolOptions>;
  /** Gradient tool options (kind / angle / stops). */
  gradientOptions: ScalarStore<GradientToolOptions>;
  /** Text tool options (font family / size / weight / line-height / align / color). */
  textOptions: ScalarStore<TextToolOptions>;
  /**
   * The active text-editing session, or `null`. React reads it to render the
   * contenteditable portal and enable the text options bar's live-restyle path;
   * the engine reads it to skip the session layer in the composite. Cleared on
   * commit, cancel, real tool switch, layer delete, or document replace.
   */
  textEditSession: ScalarStore<TextEditSession | null>;
  /**
   * The live shape-tool drag preview (document-space rect + kind), or `null`
   * when idle. The overlay renders the shape outline in place of a committed
   * layer so the drag tracks without dispatching; cleared on commit/cancel.
   */
  shapePreview: ScalarStore<{ rect: Rect; kind: 'rect' | 'ellipse' } | null>;
  /**
   * The live gradient-tool drag preview: the drag vector's start/end points in
   * document space, drawn on the overlay as a direction indicator (a gradient
   * necessarily fills the document, so only its ANGLE is previewed, not a
   * bounding box). `null` when idle; cleared on commit/cancel.
   */
  gradientPreview: ScalarStore<{ start: Vec2; end: Vec2 } | null>;
  /**
   * Whether a pixel selection currently exists. React reads it to enable the
   * fill/erase/invert/deselect controls and the engine gates selection hotkeys
   * and marching-ants animation off it. The engine writes it as the selection
   * mask gains/loses content.
   */
  hasSelection: ScalarStore<boolean>;
  /** Core-only visual SAM interaction state; application session status remains outside the engine. */
  samInteraction: ScalarStore<SamInteractionState | null>;
  /**
   * The in-progress lasso polygon (document-space points) during a lasso drag,
   * or `null` when idle. The overlay renders it as a live dashed preview in
   * place of a committed selection; cleared on commit/cancel. Like `bboxPreview`,
   * it is a transient channel — no dispatch, no React subscriber.
   */
  lassoPreview: ScalarStore<readonly Vec2[] | null>;
  /** Model-dependent grid size (document px) the bbox snaps to. React feeds this from generate settings. */
  bboxGrid: ScalarStore<number>;
  /**
   * The live bbox preview rect during a bbox-tool gesture (document space), or
   * `null` when idle. The overlay renders this in place of the committed bbox so
   * the frame tracks the drag without dispatching; cleared on commit/cancel.
   */
  bboxPreview: ScalarStore<Rect | null>;
  /**
   * The active transform-tool session (layer id + start/live transform), or
   * `null` when no session is open. React reads it to render the numeric options
   * and enable Apply/Cancel; the engine drives the live preview from it. Cleared
   * on Apply, Cancel, tool switch, or document replace.
   */
  transformSession: ScalarStore<TransformSession | null>;
  /**
   * Whether the transparency checkerboard is drawn behind transparent documents
   * (default ON). Off shows the widget surface through the document instead. The
   * compositor reads this each frame; the canvas settings menu toggles it.
   */
  checkerboard: ScalarStore<boolean>;
  /**
   * The two square colors of the transparency checkerboard, resolved from Chakra
   * semantic tokens in React and fed down (see `widgets/canvas/checkerColors.ts`).
   * The engine rebuilds its cached checker tile and recomposites when these
   * change (e.g. a theme/color-mode switch); {@link DEFAULT_CHECKER_COLORS} is the
   * React-free fallback until the first feed.
   */
  checkerColors: ScalarStore<CheckerColors>;
  /**
   * Whether the document-space grid (at the bbox snap size) is drawn on the
   * overlay (default OFF). The overlay reads this each frame; the canvas settings
   * menu toggles it.
   */
  showGrid: ScalarStore<boolean>;
  /**
   * Whether ctrl+wheel brush/eraser sizing is inverted (default OFF): normally
   * wheel-up grows the size. The wheel handler reads this; the canvas settings
   * menu toggles it. Purely an input preference — no render effect.
   */
  invertBrushSizeScroll: ScalarStore<boolean>;
  /**
   * Whether the generation bbox (dashed frame) is drawn as passive overlay chrome
   * (default ON). The overlay reads this each frame; the canvas settings popover
   * toggles it. The bbox is still drawn (with its handles) while the bbox TOOL is
   * active regardless, so it stays editable — this only hides the passive frame.
   */
  showBbox: ScalarStore<boolean>;
  /**
   * Whether the bbox overlay shade is drawn (default OFF): a translucent dark
   * fill over everything OUTSIDE the bbox, focusing attention on the generation
   * region (legacy `CanvasBboxToolModule` overlayRect parity). Overlay-only —
   * toggling never recomposites the document.
   */
  bboxOverlay: ScalarStore<boolean>;
  /**
   * Whether the rule-of-thirds composition guide (two vertical + two horizontal
   * lines dividing the bbox into thirds) is drawn inside the bbox (default OFF).
   * The overlay reads this each frame; the canvas settings popover toggles it.
   */
  ruleOfThirds: ScalarStore<boolean>;
  /**
   * Whether bbox-tool moves/resizes snap to the model grid (default ON). The bbox
   * tool reads this; the canvas settings popover toggles it, and the fit-bbox
   * header actions honor it too. Holding Alt bypasses snapping independently.
   * Purely an interaction preference — no render effect.
   */
  snapToGrid: ScalarStore<boolean>;
  /** Whether an engine-owned operation currently excludes ordinary document edits. */
  documentEditingLocked: ScalarStore<boolean>;
}

const brushOptionsEqual = (a: BrushOptions, b: BrushOptions): boolean =>
  a.size === b.size &&
  a.color === b.color &&
  a.opacity === b.opacity &&
  a.pressureSensitivity === b.pressureSensitivity;

const eraserOptionsEqual = (a: EraserOptions, b: EraserOptions): boolean =>
  a.size === b.size && a.opacity === b.opacity;

const checkerColorsEqual = (a: CheckerColors, b: CheckerColors): boolean => a.a === b.a && a.b === b.b;

const bboxOptionsEqual = (a: BboxToolOptions, b: BboxToolOptions): boolean =>
  a.aspectLocked === b.aspectLocked && a.aspectRatio === b.aspectRatio;

const lassoOptionsEqual = (a: LassoToolOptions, b: LassoToolOptions): boolean => a.mode === b.mode;

const shapeOptionsEqual = (a: ShapeToolOptions, b: ShapeToolOptions): boolean =>
  a.kind === b.kind && a.fill === b.fill && a.stroke === b.stroke && a.strokeWidth === b.strokeWidth;

const stopsEqual = (a: readonly GradientStop[], b: readonly GradientStop[]): boolean =>
  a.length === b.length && a.every((stop, i) => stop.offset === b[i]?.offset && stop.color === b[i]?.color);

const gradientOptionsEqual = (a: GradientToolOptions, b: GradientToolOptions): boolean =>
  a.kind === b.kind && a.angle === b.angle && stopsEqual(a.stops, b.stops);

const shapePreviewEqual = (
  a: { rect: Rect; kind: 'rect' | 'ellipse' } | null,
  b: { rect: Rect; kind: 'rect' | 'ellipse' } | null
): boolean => {
  if (a === null || b === null) {
    return a === b;
  }
  return (
    a.kind === b.kind &&
    a.rect.x === b.rect.x &&
    a.rect.y === b.rect.y &&
    a.rect.width === b.rect.width &&
    a.rect.height === b.rect.height
  );
};

const gradientPreviewEqual = (a: { start: Vec2; end: Vec2 } | null, b: { start: Vec2; end: Vec2 } | null): boolean => {
  if (a === null || b === null) {
    return a === b;
  }
  return a.start.x === b.start.x && a.start.y === b.start.y && a.end.x === b.end.x && a.end.y === b.end.y;
};

const bboxPreviewEqual = (a: Rect | null, b: Rect | null): boolean => {
  if (a === null || b === null) {
    return a === b;
  }
  return a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;
};

const transformEqual = (a: LayerTransform, b: LayerTransform): boolean =>
  a.x === b.x && a.y === b.y && a.scaleX === b.scaleX && a.scaleY === b.scaleY && a.rotation === b.rotation;

const transformSessionEqual = (a: TransformSession | null, b: TransformSession | null): boolean => {
  if (a === null || b === null) {
    return a === b;
  }
  return (
    a.layerId === b.layerId &&
    transformEqual(a.startTransform, b.startTransform) &&
    transformEqual(a.transform, b.transform)
  );
};

const textOptionsEqual = (a: TextToolOptions, b: TextToolOptions): boolean =>
  a.fontFamily === b.fontFamily &&
  a.fontSize === b.fontSize &&
  a.fontWeight === b.fontWeight &&
  a.lineHeight === b.lineHeight &&
  a.align === b.align &&
  a.color === b.color;

const textSourceEqual = (a: TextSource, b: TextSource): boolean =>
  a.content === b.content &&
  a.fontFamily === b.fontFamily &&
  a.fontSize === b.fontSize &&
  a.fontWeight === b.fontWeight &&
  a.lineHeight === b.lineHeight &&
  a.align === b.align &&
  a.color === b.color;

const textEditSessionEqual = (a: TextEditSession | null, b: TextEditSession | null): boolean => {
  if (a === null || b === null) {
    return a === b;
  }
  return (
    a.id === b.id &&
    a.mode === b.mode &&
    a.layerId === b.layerId &&
    textSourceEqual(a.source, b.source) &&
    transformEqual(a.transform, b.transform)
  );
};

/** Creates a fresh bundle of engine stores with their initial values. */
export const createEngineStores = (initialTool: ToolId = 'view'): EngineStores => ({
  activeTool: createScalarStore<ToolId>(initialTool),
  bboxGrid: createScalarStore<number>(DEFAULT_BBOX_GRID),
  bboxOptions: createScalarStore<BboxToolOptions>({ ...DEFAULT_BBOX_OPTIONS }, bboxOptionsEqual),
  bboxPreview: createScalarStore<Rect | null>(null, bboxPreviewEqual),
  bboxOverlay: createScalarStore<boolean>(false),
  brushOptions: createScalarStore<BrushOptions>({ ...DEFAULT_BRUSH_OPTIONS }, brushOptionsEqual),
  canRedo: createScalarStore<boolean>(false),
  canUndo: createScalarStore<boolean>(false),
  checkerboard: createScalarStore<boolean>(true),
  checkerColors: createScalarStore<CheckerColors>({ ...DEFAULT_CHECKER_COLORS }, checkerColorsEqual),
  cursor: createScalarStore<string>('default'),
  eraserOptions: createScalarStore<EraserOptions>({ ...DEFAULT_ERASER_OPTIONS }, eraserOptionsEqual),
  documentEditingLocked: createScalarStore<boolean>(false),
  hasSelection: createScalarStore<boolean>(false),
  invertBrushSizeScroll: createScalarStore<boolean>(false),
  gradientOptions: createScalarStore<GradientToolOptions>(
    { ...DEFAULT_GRADIENT_OPTIONS, stops: DEFAULT_GRADIENT_OPTIONS.stops.map((s) => ({ ...s })) },
    gradientOptionsEqual
  ),
  gradientPreview: createScalarStore<{ start: Vec2; end: Vec2 } | null>(null, gradientPreviewEqual),
  lassoOptions: createScalarStore<LassoToolOptions>({ ...DEFAULT_LASSO_OPTIONS }, lassoOptionsEqual),
  lassoPreview: createScalarStore<readonly Vec2[] | null>(null),
  ruleOfThirds: createScalarStore<boolean>(false),
  samInteraction: createScalarStore(null),
  shapeOptions: createScalarStore<ShapeToolOptions>({ ...DEFAULT_SHAPE_OPTIONS }, shapeOptionsEqual),
  shapePreview: createScalarStore<{ rect: Rect; kind: 'rect' | 'ellipse' } | null>(null, shapePreviewEqual),
  showBbox: createScalarStore<boolean>(true),
  showGrid: createScalarStore<boolean>(false),
  snapToGrid: createScalarStore<boolean>(true),
  textEditSession: createScalarStore<TextEditSession | null>(null, textEditSessionEqual),
  textOptions: createScalarStore<TextToolOptions>({ ...DEFAULT_TEXT_OPTIONS }, textOptionsEqual),
  thumbnailVersion: createKeyedVersionStore(),
  thumbnailStatus: createKeyedThumbnailStatusStore(),
  transformSession: createScalarStore<TransformSession | null>(null, transformSessionEqual),
  viewportReady: createScalarStore<boolean>(false),
  zoom: createScalarStore<number>(1),
});
