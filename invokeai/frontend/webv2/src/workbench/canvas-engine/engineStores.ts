/**
 * Per-engine React-free external stores expose narrow interaction snapshots and subscriptions. Widget hooks adapt
 * them to React without coupling the engine to UI.
 */

import type {
  CanvasLayerSourceContract,
  CanvasTextFontRef,
  CanvasTextFontStyle,
  CanvasTextFontVariations,
  ParametricShapeKind,
} from '@workbench/canvas-engine/contracts';
import type { SamInteractionState } from '@workbench/canvas-engine/samInteraction';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect, SelectionOp, ToolId, Vec2 } from '@workbench/canvas-engine/types';

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
  /** Edge hardness in [0, 1]: 1 is a crisp edge, lower feathers the silhouette. */
  hardness: number;
  /** Whether pen pressure modulates the stroke width. */
  pressureAffectsWidth: boolean;
  /**
   * Pressure-alpha is off by default: unlike width modulation it refills the full scratch region each frame and
   * changes stroke appearance.
   */
  pressureAffectsOpacity: boolean;
}

/** Eraser tool options. */
export interface EraserOptions {
  /** Base eraser diameter in document units. */
  size: number;
  /** Per-stroke erase strength in [0, 1]. */
  opacity: number;
  /** Edge hardness in [0, 1]: 1 is a crisp edge, lower feathers the silhouette. */
  hardness: number;
}

/** Lasso (selection) tool options: how the path is drawn, and the op it applies. */
export interface LassoToolOptions {
  /** The op a committed lasso path applies to the selection, when no modifier overrides it. */
  mode: SelectionOp;
  /** `freehand` traces a drag; `polygon` places straight-edged vertices by click. */
  shape: 'freehand' | 'polygon';
}

/** Default lasso options: a freehand path that replaces the selection. */
export const DEFAULT_LASSO_OPTIONS: LassoToolOptions = {
  mode: 'replace',
  shape: 'freehand',
};

export interface MarqueeToolOptions {
  kind: 'rect' | 'ellipse';
  /** The op a committed marquee applies to the selection, when no modifier overrides it. */
  mode: SelectionOp;
}

/** Default marquee options: a rectangle that replaces the selection. */
export const DEFAULT_MARQUEE_OPTIONS: MarqueeToolOptions = {
  kind: 'rect',
  mode: 'replace',
};

/** A single gradient stop (offset in [0,1], any CSS color). */
export interface GradientStop {
  offset: number;
  color: string;
}

/**
 * Next-shape kind and fill/stroke toggles. Colors come from foreground/background at gesture start; selected-shape
 * colors belong to the document.
 */
export type ShapeToolKind = ParametricShapeKind | 'polygon' | 'freehand';

/** Where a drawn shape lands: pixels on the selected paint layer, or its own shape layer. */
export type ShapeToolTarget = 'selected' | 'new';

export interface ShapeToolOptions {
  /** The box-parametric kinds drag out a rect; `polygon` places vertices, `freehand` traces a drag. */
  kind: ShapeToolKind;
  /** Falls back to a new shape layer whenever the selected layer cannot take pixels. */
  target: ShapeToolTarget;
  fillEnabled: boolean;
  strokeEnabled: boolean;
  strokeWidth: number;
}

/** Sensible starting shape options: a filled rect, no stroke, drawn onto the selected paint layer. */
export const DEFAULT_SHAPE_OPTIONS: Readonly<ShapeToolOptions> = {
  fillEnabled: true,
  kind: 'rect',
  strokeEnabled: false,
  strokeWidth: 8,
  target: 'selected',
};

/** Largest shape stroke width (document px) the options bar clamps to. */
export const MAX_SHAPE_STROKE_WIDTH = 2000;

/**
 * Gradient kind, angle, preset and custom stops. Pair resolves foreground/background at gesture start; custom uses
 * stops verbatim. The editor changes first and last stops.
 */
export interface GradientToolOptions {
  kind: 'linear' | 'radial';
  angle: number;
  /** `pair` = the built-in FG→BG preset, resolved when the drag starts. */
  preset: 'pair' | 'custom';
  stops: GradientStop[];
}

/** The lasso tool's in-flight outline: a freehand drag, or a polygon being placed vertex by vertex. */
export type LassoPreview =
  | { kind: 'freehand'; points: readonly Vec2[] }
  | {
      kind: 'polygon';
      points: readonly Vec2[];
      cursor: Vec2 | null;
      /** Once the polygon can close, a press this many screen px from the first vertex closes it; the overlay rings that radius. */
      closeRadiusPx: number | null;
      closeArmed: boolean;
    };

/** The gradient tool's in-flight drag: start → end in document space. */
export interface GradientPreview {
  kind: 'linear' | 'radial';
  start: Vec2;
  end: Vec2;
}

/** Sensible starting gradient options: the FG→BG preset, horizontal linear. */
export const DEFAULT_GRADIENT_OPTIONS: Readonly<Omit<GradientToolOptions, 'stops'>> & {
  readonly stops: readonly Readonly<GradientStop>[];
} = {
  angle: 0,
  kind: 'linear',
  preset: 'pair',
  stops: [
    { color: '#000000ff', offset: 0 },
    { color: '#00000000', offset: 1 },
  ],
};

/**
 * New-text defaults and live style. Initial color comes from foreground; active session and selected-layer colors
 * are independently stored.
 */
export interface TextToolOptions {
  fontFamily: string;
  fontSize: number;
  /** CSS numeric weight (400/500/600/700). */
  fontWeight: number;
  /** Stable custom-font identity, omitted for built-in font stacks. */
  fontRef?: CanvasTextFontRef;
  /** CSS style used by both the editing portal and Canvas rasterizer. Defaults to `normal`. */
  fontStyle?: CanvasTextFontStyle;
  /** Explicit OpenType variation coordinates for custom or variable faces. Defaults to `{}`. */
  fontVariations?: CanvasTextFontVariations;
  /** Unitless line-height multiplier over `fontSize`. */
  lineHeight: number;
  align: 'left' | 'center' | 'right';
}

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

/** What a live text session (or a selected text layer) can restyle: the options plus its explicit color. */
export type TextStylePatch = Partial<TextToolOptions> & { color?: string };

/** Sensible starting text options: left-aligned Inter at 48px. */
export const DEFAULT_TEXT_OPTIONS: Readonly<TextToolOptions> = {
  align: 'left',
  fontFamily: TEXT_FONT_FAMILIES[0]!.value,
  fontSize: 48,
  fontWeight: 400,
  fontStyle: 'normal',
  fontVariations: {},
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
export const MIN_BRUSH_SIZE = 0.1;
export const MAX_BRUSH_SIZE = 2000;

/** The mirrored foreground/background pair the object tools read at gesture start. */
export interface ActiveColorPairState {
  foreground: string;
  background: string;
}

/** Matches the workbench pair's default: black on white. */
export const DEFAULT_COLOR_PAIR_STATE: ActiveColorPairState = { background: '#ffffff', foreground: '#000000' };

/** Sensible starting brush options. */
export const DEFAULT_BRUSH_OPTIONS: Readonly<BrushOptions> = {
  color: '#000000',
  hardness: 1,
  opacity: 1,
  pressureAffectsOpacity: false,
  pressureAffectsWidth: true,
  size: 50,
};

/** Sensible starting eraser options. */
export const DEFAULT_ERASER_OPTIONS: Readonly<EraserOptions> = {
  hardness: 1,
  opacity: 1,
  size: 50,
};

/**
 * Transform sessions span gestures until Apply/Cancel. `startTransform` is the committed undo/cancel baseline;
 * `transform` drives preview and numeric controls.
 */
export interface TransformSession {
  layerId: string;
  startTransform: LayerTransform;
  transform: LayerTransform;
}

/**
 * Text sessions suppress the edited layer while its portal renders. Create has no layer/start source until one add
 * commit; edit keeps `startSource` as undo and no-change baseline for one source-update commit.
 *
 * `source` holds live styles but only seeded content; typing stays in the DOM until commit. `transform` places the
 * portal, and incrementing `id` remounts it per session.
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

/** Per-key subscriptions rerender only the affected thumbnail; global subscription supports coarse observers. */
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
  /** Monotonic aggregate version for consumers whose eligibility depends on live layer content. */
  layerPixelEpoch: ScalarStore<number>;
  /** Monotonic version of the mirrored document, for consumers that read the engine's model during render. */
  documentEpoch: ScalarStore<number>;
  /** Brush tool options (size / color / opacity / pressure). */
  brushOptions: ScalarStore<BrushOptions>;
  /** One-way foreground/background mirror for gesture-start reads; the engine never writes it. */
  colorPair: ScalarStore<ActiveColorPairState>;
  /** Eraser tool options (size / opacity). */
  eraserOptions: ScalarStore<EraserOptions>;
  /** Whether the engine-owned canvas history has an entry to undo. */
  canUndo: ScalarStore<boolean>;
  /** Whether the engine-owned canvas history has an entry to redo. */
  canRedo: ScalarStore<boolean>;
  /** Bumped on every history-stack mutation, so a history list can re-read the entries. */
  historyEpoch: ScalarStore<number>;
  /** Bbox tool options (aspect lock / ratio). */
  bboxOptions: ScalarStore<BboxToolOptions>;
  /** Lasso tool options (the committed boolean op mode). */
  lassoOptions: ScalarStore<LassoToolOptions>;
  /** Marquee tool options (shape kind / the committed boolean op mode). */
  marqueeOptions: ScalarStore<MarqueeToolOptions>;
  /** Shape tool options (kind / fill and stroke enablement / stroke width). */
  shapeOptions: ScalarStore<ShapeToolOptions>;
  /** Gradient tool options (kind / angle / preset / stops). */
  gradientOptions: ScalarStore<GradientToolOptions>;
  /** Text tool options (font family / size / weight / line-height / align). */
  textOptions: ScalarStore<TextToolOptions>;
  /**
   * Text session drives the portal, live style controls and composite suppression. Commit, cancel, real tool
   * switch, deletion or replacement clears it.
   */
  textEditSession: ScalarStore<TextEditSession | null>;
  /** Document-space shape preview replaces dispatch during drag; null when idle, cleared on commit/cancel. */
  shapePreview: ScalarStore<{ rect: Rect; kind: ParametricShapeKind } | null>;
  /** Document-space gradient drag vector; overlay shows a linear ramp or radial circle. Cleared on commit/cancel. */
  gradientPreview: ScalarStore<GradientPreview | null>;
  hasSelection: ScalarStore<boolean>;
  /** Floating pixels enable Apply/Cancel without a separate transform session. */
  hasFloatingSelection: ScalarStore<boolean>;
  /** Core-only visual SAM interaction state; application session status remains outside the engine. */
  samInteraction: ScalarStore<SamInteractionState | null>;
  /**
   * Transient lasso overlay: freehand outline or polygon vertices, cursor band and close target. Cleared on
   * commit/cancel without dispatch.
   */
  lassoPreview: ScalarStore<LassoPreview | null>;
  /** Transient marquee outline; the selection mask changes only on commit. */
  marqueePreview: ScalarStore<{ rect: Rect; kind: 'rect' | 'ellipse' } | null>;
  /** Legacy bbox stroke clipping, captured once per paint gesture. */
  clipToBbox: ScalarStore<boolean>;
  /** Model-dependent grid size (document px) the bbox snaps to. React feeds this from generate settings. */
  bboxGrid: ScalarStore<number>;
  /**
   * Document-space bbox preview replaces the committed overlay frame during drag without dispatch. Cleared on
   * commit/cancel.
   */
  bboxPreview: ScalarStore<Rect | null>;
  /** Transform session supplies numeric controls, Apply/Cancel and live preview; null when closed. */
  transformSession: ScalarStore<TransformSession | null>;
  /** Checkerboard defaults on; disabling reveals the widget surface through transparent document pixels. */
  checkerboard: ScalarStore<boolean>;
  /**
   * React supplies semantic checker colors; changes rebuild the tile and recompose. {@link DEFAULT_CHECKER_COLORS}
   * supplies the React-free initial fallback.
   */
  checkerColors: ScalarStore<CheckerColors>;
  showGrid: ScalarStore<boolean>;
  /** Inverts ctrl+wheel sizing; default wheel-up grows. Input-only preference. */
  invertBrushSizeScroll: ScalarStore<boolean>;
  /** Passive bbox frame defaults on. The active bbox tool always draws its editable frame and handles. */
  showBbox: ScalarStore<boolean>;
  /** Optional shade outside the generation bbox. Overlay-only toggling never recomposites document pixels. */
  bboxOverlay: ScalarStore<boolean>;
  ruleOfThirds: ScalarStore<boolean>;
  /** Bbox moves, resizes and fit actions use grid snapping by default; Alt bypasses it. Input-only preference. */
  snapToGrid: ScalarStore<boolean>;
  /** Whether an engine-owned operation currently excludes ordinary document edits. */
  documentEditingLocked: ScalarStore<boolean>;
  /** The layer targeted by that operation, or null when no operation is active. */
  documentEditingLayerId: ScalarStore<string | null>;
}

const brushOptionsEqual = (a: BrushOptions, b: BrushOptions): boolean =>
  a.size === b.size &&
  a.color === b.color &&
  a.opacity === b.opacity &&
  a.hardness === b.hardness &&
  a.pressureAffectsWidth === b.pressureAffectsWidth &&
  a.pressureAffectsOpacity === b.pressureAffectsOpacity;

const eraserOptionsEqual = (a: EraserOptions, b: EraserOptions): boolean =>
  a.size === b.size && a.opacity === b.opacity && a.hardness === b.hardness;

const checkerColorsEqual = (a: CheckerColors, b: CheckerColors): boolean => a.a === b.a && a.b === b.b;

const bboxOptionsEqual = (a: BboxToolOptions, b: BboxToolOptions): boolean =>
  a.aspectLocked === b.aspectLocked && a.aspectRatio === b.aspectRatio;

const lassoOptionsEqual = (a: LassoToolOptions, b: LassoToolOptions): boolean =>
  a.mode === b.mode && a.shape === b.shape;

const marqueeOptionsEqual = (a: MarqueeToolOptions, b: MarqueeToolOptions): boolean =>
  a.kind === b.kind && a.mode === b.mode;

const shapeOptionsEqual = (a: ShapeToolOptions, b: ShapeToolOptions): boolean =>
  a.kind === b.kind &&
  a.target === b.target &&
  a.fillEnabled === b.fillEnabled &&
  a.strokeEnabled === b.strokeEnabled &&
  a.strokeWidth === b.strokeWidth;

const colorPairEqual = (a: ActiveColorPairState, b: ActiveColorPairState): boolean =>
  a.foreground === b.foreground && a.background === b.background;

const stopsEqual = (a: readonly GradientStop[], b: readonly GradientStop[]): boolean =>
  a.length === b.length && a.every((stop, i) => stop.offset === b[i]?.offset && stop.color === b[i]?.color);

const gradientOptionsEqual = (a: GradientToolOptions, b: GradientToolOptions): boolean =>
  a.kind === b.kind && a.angle === b.angle && a.preset === b.preset && stopsEqual(a.stops, b.stops);

/** Shared by the shape and marquee previews — both are a rect plus a shape kind. */
const rectShapePreviewEqual = (
  a: { rect: Rect; kind: ParametricShapeKind } | null,
  b: { rect: Rect; kind: ParametricShapeKind } | null
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

const gradientPreviewEqual = (a: GradientPreview | null, b: GradientPreview | null): boolean => {
  if (a === null || b === null) {
    return a === b;
  }
  return (
    a.kind === b.kind &&
    a.start.x === b.start.x &&
    a.start.y === b.start.y &&
    a.end.x === b.end.x &&
    a.end.y === b.end.y
  );
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

const textFontRefEqual = (a: CanvasTextFontRef | undefined, b: CanvasTextFontRef | undefined): boolean =>
  a === b ||
  (a !== undefined &&
    b !== undefined &&
    a.id === b.id &&
    a.contentHash === b.contentHash &&
    a.family === b.family &&
    a.label === b.label);

const textVariationsEqual = (
  a: CanvasTextFontVariations | undefined,
  b: CanvasTextFontVariations | undefined
): boolean => {
  const aKeys = Object.keys(a ?? {});
  const bKeys = Object.keys(b ?? {});
  return aKeys.length === bKeys.length && aKeys.every((key) => (a ?? {})[key] === (b ?? {})[key]);
};

const textOptionsEqual = (a: TextToolOptions, b: TextToolOptions): boolean =>
  a.fontFamily === b.fontFamily &&
  a.fontSize === b.fontSize &&
  a.fontWeight === b.fontWeight &&
  textFontRefEqual(a.fontRef, b.fontRef) &&
  a.fontStyle === b.fontStyle &&
  textVariationsEqual(a.fontVariations, b.fontVariations) &&
  a.lineHeight === b.lineHeight &&
  a.align === b.align;

const textSourceEqual = (a: TextSource, b: TextSource): boolean =>
  a.content === b.content &&
  a.fontFamily === b.fontFamily &&
  a.fontSize === b.fontSize &&
  a.fontWeight === b.fontWeight &&
  textFontRefEqual(a.fontRef, b.fontRef) &&
  (a.fontStyle ?? 'normal') === (b.fontStyle ?? 'normal') &&
  textVariationsEqual(a.fontVariations ?? {}, b.fontVariations ?? {}) &&
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
  colorPair: createScalarStore<ActiveColorPairState>({ ...DEFAULT_COLOR_PAIR_STATE }, colorPairEqual),
  canRedo: createScalarStore<boolean>(false),
  canUndo: createScalarStore<boolean>(false),
  historyEpoch: createScalarStore<number>(0),
  checkerboard: createScalarStore<boolean>(true),
  clipToBbox: createScalarStore<boolean>(false),
  checkerColors: createScalarStore<CheckerColors>({ ...DEFAULT_CHECKER_COLORS }, checkerColorsEqual),
  cursor: createScalarStore<string>('default'),
  eraserOptions: createScalarStore<EraserOptions>({ ...DEFAULT_ERASER_OPTIONS }, eraserOptionsEqual),
  documentEditingLayerId: createScalarStore<string | null>(null),
  documentEditingLocked: createScalarStore<boolean>(false),
  hasFloatingSelection: createScalarStore<boolean>(false),
  hasSelection: createScalarStore<boolean>(false),
  invertBrushSizeScroll: createScalarStore<boolean>(false),
  gradientOptions: createScalarStore<GradientToolOptions>(
    { ...DEFAULT_GRADIENT_OPTIONS, stops: DEFAULT_GRADIENT_OPTIONS.stops.map((s) => ({ ...s })) },
    gradientOptionsEqual
  ),
  gradientPreview: createScalarStore<GradientPreview | null>(null, gradientPreviewEqual),
  lassoOptions: createScalarStore<LassoToolOptions>({ ...DEFAULT_LASSO_OPTIONS }, lassoOptionsEqual),
  lassoPreview: createScalarStore<LassoPreview | null>(null),
  marqueeOptions: createScalarStore<MarqueeToolOptions>({ ...DEFAULT_MARQUEE_OPTIONS }, marqueeOptionsEqual),
  marqueePreview: createScalarStore<{ rect: Rect; kind: 'rect' | 'ellipse' } | null>(null, rectShapePreviewEqual),
  layerPixelEpoch: createScalarStore<number>(0),
  documentEpoch: createScalarStore<number>(0),
  ruleOfThirds: createScalarStore<boolean>(false),
  samInteraction: createScalarStore(null),
  shapeOptions: createScalarStore<ShapeToolOptions>({ ...DEFAULT_SHAPE_OPTIONS }, shapeOptionsEqual),
  shapePreview: createScalarStore<{ rect: Rect; kind: ParametricShapeKind } | null>(null, rectShapePreviewEqual),
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
