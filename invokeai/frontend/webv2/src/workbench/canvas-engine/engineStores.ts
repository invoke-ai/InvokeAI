import type { PsdExportStatus } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasLayerSourceContract,
  CanvasTextFontRef,
  CanvasTextFontStyle,
  CanvasTextFontVariations,
  ParametricShapeKind,
} from '@workbench/canvas-engine/contracts';
import type { AdjustedSurfaceStatus } from '@workbench/canvas-engine/render/spatialDisplay';
import type { VectorShapePreview } from '@workbench/canvas-engine/render/vectorGuides';
import type { SamInteractionState } from '@workbench/canvas-engine/samInteraction';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect, SelectionOp, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { VectorEditSession } from '@workbench/canvas-engine/vectorEditState';

import { type CheckerColors, DEFAULT_CHECKER_COLORS } from '@workbench/canvas-engine/render/compositor';

export type { CheckerColors };

export type TextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;

export interface BrushOptions {
  /** Document units. */
  size: number;
  color: string;
  opacity: number;
  /** 1 is a crisp edge; lower feathers the silhouette. */
  hardness: number;
  pressureAffectsWidth: boolean;
  /** Off by default: it refills the full scratch region each frame and changes stroke appearance. */
  pressureAffectsOpacity: boolean;
}

export interface EraserOptions {
  /** Document units. */
  size: number;
  opacity: number;
  /** 1 is a crisp edge; lower feathers the silhouette. */
  hardness: number;
  /** Pressure never modulates erase strength. */
  pressureAffectsWidth: boolean;
}

/** Q quick-switch memory; transient, it lives with the engine, never in the project. */
export interface LayerQuickSwitchState {
  /** Readers ignore a stale id. */
  bookmarkedLayerId: string | null;
  /** Engine-maintained; internal edits and deselection preserve it. */
  previousLayerId: string | null;
}

export interface LassoToolOptions {
  /** Applies when no modifier overrides it. */
  mode: SelectionOp;
  shape: 'freehand' | 'polygon';
}

export const DEFAULT_LASSO_OPTIONS: LassoToolOptions = {
  mode: 'replace',
  shape: 'freehand',
};

export interface MarqueeToolOptions {
  kind: 'rect' | 'ellipse';
  /** Applies when no modifier overrides it. */
  mode: SelectionOp;
}

export const DEFAULT_MARQUEE_OPTIONS: MarqueeToolOptions = {
  kind: 'rect',
  mode: 'replace',
};

/** Offset in [0, 1]. */
export interface GradientStop {
  offset: number;
  color: string;
}

export type ShapeToolKind = ParametricShapeKind | 'polygon' | 'freehand';

export type ShapeToolTarget = 'selected' | 'new';

/** Colors come from the color pair at gesture start; selected-shape colors belong to the document. */
export interface ShapeToolOptions {
  kind: ShapeToolKind;
  /** Falls back to a new shape layer whenever the selected layer cannot take pixels. */
  target: ShapeToolTarget;
  fillEnabled: boolean;
  strokeEnabled: boolean;
  strokeWidth: number;
}

export const DEFAULT_SHAPE_OPTIONS: Readonly<ShapeToolOptions> = {
  fillEnabled: true,
  kind: 'rect',
  strokeEnabled: false,
  strokeWidth: 8,
  target: 'selected',
};

export const MAX_SHAPE_STROKE_WIDTH = 2000;

/** `custom` uses `stops` verbatim. */
export interface GradientToolOptions {
  kind: 'linear' | 'radial';
  angle: number;
  /** `pair` is the FG→BG preset, resolved when the drag starts. */
  preset: 'pair' | 'custom';
  stops: GradientStop[];
}

export type LassoPreview =
  | { kind: 'freehand'; points: readonly Vec2[] }
  | {
      kind: 'polygon';
      points: readonly Vec2[];
      cursor: Vec2 | null;
      /** Screen px from the first vertex that closes the polygon; null until it can close. */
      closeRadiusPx: number | null;
      closeArmed: boolean;
    };

export interface GradientPreview {
  kind: 'linear' | 'radial';
  start: Vec2;
  end: Vec2;
}

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

/** Initial color is the foreground; session and selected-layer colors are stored apart. */
export interface TextToolOptions {
  fontFamily: string;
  fontSize: number;
  fontWeight: number;
  /** Omitted for built-in font stacks. */
  fontRef?: CanvasTextFontRef;
  /** Defaults to `normal`. */
  fontStyle?: CanvasTextFontStyle;
  /** Defaults to `{}`. */
  fontVariations?: CanvasTextFontVariations;
  /** Unitless multiplier over `fontSize`. */
  lineHeight: number;
  align: 'left' | 'center' | 'right';
}

export const TEXT_FONT_FAMILIES: readonly { label: string; value: string }[] = [
  { label: 'Inter', value: "'Inter Variable', Inter, sans-serif" },
  { label: 'Sans-serif', value: 'system-ui, sans-serif' },
  { label: 'Serif', value: "Georgia, 'Times New Roman', serif" },
  { label: 'Monospace', value: "'JetBrains Mono', ui-monospace, monospace" },
];

export const TEXT_FONT_WEIGHTS: readonly number[] = [400, 500, 600, 700];

/** Document px. */
export const MIN_TEXT_FONT_SIZE = 1;
export const MAX_TEXT_FONT_SIZE = 2000;

export type TextStylePatch = Partial<TextToolOptions> & { color?: string };

export const DEFAULT_TEXT_OPTIONS: Readonly<TextToolOptions> = {
  align: 'left',
  fontFamily: TEXT_FONT_FAMILIES[0]!.value,
  fontSize: 48,
  fontWeight: 400,
  fontStyle: 'normal',
  fontVariations: {},
  lineHeight: 1.2,
};

export interface BboxToolOptions {
  aspectLocked: boolean;
  aspectRatio: number;
}

/** Used until a model feeds a real grid. */
export const DEFAULT_BBOX_GRID = 8;

export const DEFAULT_BBOX_OPTIONS: BboxToolOptions = {
  aspectLocked: false,
  aspectRatio: 1,
};

/** Document units. */
export const MIN_BRUSH_SIZE = 0.1;
export const MAX_BRUSH_SIZE = 2000;

export interface ActiveColorPairState {
  foreground: string;
  background: string;
}

export const DEFAULT_COLOR_PAIR_STATE: ActiveColorPairState = { background: '#ffffff', foreground: '#000000' };

export const DEFAULT_BRUSH_OPTIONS: Readonly<BrushOptions> = {
  color: '#000000',
  hardness: 1,
  opacity: 1,
  pressureAffectsOpacity: false,
  pressureAffectsWidth: true,
  size: 50,
};

export const DEFAULT_ERASER_OPTIONS: Readonly<EraserOptions> = {
  hardness: 1,
  opacity: 1,
  pressureAffectsWidth: true,
  size: 50,
};

export interface ControlLayerView {
  layerId: string;
  view: 'original' | 'overlay';
}

export interface TransformOptions {
  smoothing: boolean;
  resampleMode: 'bilinear' | 'bicubic' | 'hamming' | 'lanczos';
  fitMode: 'contain' | 'cover' | 'fill';
}

/** Spans gestures until Apply or Cancel; `startTransform` is the undo and cancel baseline. */
export interface TransformSession {
  processing?: boolean;
  processingPhase?: 'preparing' | 'queued' | 'resampling';
  applyFailed?: boolean;
  layerId: string;
  startTransform: LayerTransform;
  transform: LayerTransform;
}

/** Create has no layer until its add commits; `source` holds live styles but typing stays in the DOM until commit. */
export interface TextEditSession {
  id: number;
  mode: 'create' | 'edit';
  layerId: string | null;
  startSource: TextSource | null;
  source: TextSource;
  transform: LayerTransform;
}

/** `useSyncExternalStore`-compatible. */
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

/** Per-key subscriptions rerender only the affected thumbnail. */
export interface KeyedVersionStore {
  get(key: string): number | undefined;
  set(key: string, value: number): void;
  delete(key: string): void;
  subscribeKey(key: string, listener: () => void): () => void;
  subscribe(listener: () => void): () => void;
}

export type LayerThumbnailStatus = 'loading' | 'ready' | 'error';

/** Absent means exact, or no adjustments. */
export type LayerAdjustmentStatus = AdjustedSurfaceStatus;

/** Absent means `idle`. */
export interface KeyedStatusStore<T extends string> {
  get(key: string): T | undefined;
  set(key: string, value: T): void;
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

const createKeyedStatusStore = <T extends string>(): KeyedStatusStore<T> => {
  const values = new Map<string, T>();
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

export interface EngineStores {
  activeTool: ScalarStore<ToolId>;
  zoom: ScalarStore<number>;
  viewportReady: ScalarStore<boolean>;
  cursor: ScalarStore<string>;
  thumbnailVersion: KeyedVersionStore;
  thumbnailStatus: KeyedStatusStore<LayerThumbnailStatus>;
  adjustmentStatus: KeyedStatusStore<LayerAdjustmentStatus>;
  layerPixelEpoch: ScalarStore<number>;
  documentEpoch: ScalarStore<number>;
  brushOptions: ScalarStore<BrushOptions>;
  /** One-way mirror; the engine never writes it. */
  colorPair: ScalarStore<ActiveColorPairState>;
  eraserOptions: ScalarStore<EraserOptions>;
  layerQuickSwitch: ScalarStore<LayerQuickSwitchState>;
  canUndo: ScalarStore<boolean>;
  canRedo: ScalarStore<boolean>;
  historyEpoch: ScalarStore<number>;
  bboxOptions: ScalarStore<BboxToolOptions>;
  lassoOptions: ScalarStore<LassoToolOptions>;
  marqueeOptions: ScalarStore<MarqueeToolOptions>;
  shapeOptions: ScalarStore<ShapeToolOptions>;
  gradientOptions: ScalarStore<GradientToolOptions>;
  textOptions: ScalarStore<TextToolOptions>;
  /** Cleared by commit, cancel, a real tool switch, deletion or replacement. */
  textEditSession: ScalarStore<TextEditSession | null>;
  shapePreview: ScalarStore<{ rect: Rect; kind: ParametricShapeKind } | null>;
  /** Vector creation preview is guide geometry, never a rasterization source. */
  vectorShapePreview: ScalarStore<VectorShapePreview | null>;
  vectorEditSession: ScalarStore<VectorEditSession | null>;
  vectorToolsStatus: ScalarStore<'idle' | 'loading' | 'ready' | 'failed'>;
  /** Document-space gradient drag vector; overlay shows a linear ramp or radial circle. Cleared on commit/cancel. */
  gradientPreview: ScalarStore<GradientPreview | null>;
  hasSelection: ScalarStore<boolean>;
  hasFloatingSelection: ScalarStore<boolean>;
  /** Application session status stays outside the engine. */
  samInteraction: ScalarStore<SamInteractionState | null>;
  lassoPreview: ScalarStore<LassoPreview | null>;
  /** The selection mask changes only on commit. */
  marqueePreview: ScalarStore<{ rect: Rect; kind: 'rect' | 'ellipse' } | null>;
  /** Captured once per paint gesture. */
  clipToBbox: ScalarStore<boolean>;
  /** React feeds this from generate settings. */
  bboxGrid: ScalarStore<number>;
  bboxPreview: ScalarStore<Rect | null>;
  /** Transient; not part of the persisted document. */
  transformOptions: ScalarStore<TransformOptions>;
  transformSession: ScalarStore<TransformSession | null>;
  pendingTransformLayerId: ScalarStore<string | null>;
  checkerboard: ScalarStore<boolean>;
  /** {@link DEFAULT_CHECKER_COLORS} is the React-free fallback. */
  checkerColors: ScalarStore<CheckerColors>;
  showGrid: ScalarStore<boolean>;
  /** Default wheel-up grows. */
  invertBrushSizeScroll: ScalarStore<boolean>;
  /** The active bbox tool always draws its frame and handles. */
  showBbox: ScalarStore<boolean>;
  /** Toggling never recomposites document pixels. */
  bboxOverlay: ScalarStore<boolean>;
  ruleOfThirds: ScalarStore<boolean>;
  /** Alt bypasses it. */
  snapToGrid: ScalarStore<boolean>;
  documentEditingLocked: ScalarStore<boolean>;
  psdExportStatus: ScalarStore<PsdExportStatus>;

  documentEditingLayerId: ScalarStore<string | null>;
  controlLayerView: ScalarStore<ControlLayerView | null>;
  stagingHidesOverlays: ScalarStore<boolean>;
}

const brushOptionsEqual = (a: BrushOptions, b: BrushOptions): boolean =>
  a.size === b.size &&
  a.color === b.color &&
  a.opacity === b.opacity &&
  a.hardness === b.hardness &&
  a.pressureAffectsWidth === b.pressureAffectsWidth &&
  a.pressureAffectsOpacity === b.pressureAffectsOpacity;

const eraserOptionsEqual = (a: EraserOptions, b: EraserOptions): boolean =>
  a.size === b.size &&
  a.opacity === b.opacity &&
  a.hardness === b.hardness &&
  a.pressureAffectsWidth === b.pressureAffectsWidth;

const layerQuickSwitchEqual = (a: LayerQuickSwitchState, b: LayerQuickSwitchState): boolean =>
  a.bookmarkedLayerId === b.bookmarkedLayerId && a.previousLayerId === b.previousLayerId;

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
    a.processing === b.processing &&
    a.processingPhase === b.processingPhase &&
    a.applyFailed === b.applyFailed &&
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
  controlLayerView: createScalarStore<ControlLayerView | null>(null),
  cursor: createScalarStore<string>('default'),
  eraserOptions: createScalarStore<EraserOptions>({ ...DEFAULT_ERASER_OPTIONS }, eraserOptionsEqual),
  documentEditingLayerId: createScalarStore<string | null>(null),
  documentEditingLocked: createScalarStore<boolean>(false),
  psdExportStatus: createScalarStore<PsdExportStatus>('idle'),
  hasFloatingSelection: createScalarStore<boolean>(false),
  hasSelection: createScalarStore<boolean>(false),
  invertBrushSizeScroll: createScalarStore<boolean>(false),
  layerQuickSwitch: createScalarStore<LayerQuickSwitchState>(
    { bookmarkedLayerId: null, previousLayerId: null },
    layerQuickSwitchEqual
  ),
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
  vectorShapePreview: createScalarStore<VectorShapePreview | null>(null),
  vectorEditSession: createScalarStore<VectorEditSession | null>(null),
  vectorToolsStatus: createScalarStore<'idle' | 'loading' | 'ready' | 'failed'>('idle'),
  showBbox: createScalarStore<boolean>(true),
  showGrid: createScalarStore<boolean>(false),
  snapToGrid: createScalarStore<boolean>(true),
  stagingHidesOverlays: createScalarStore<boolean>(false),
  textEditSession: createScalarStore<TextEditSession | null>(null, textEditSessionEqual),
  textOptions: createScalarStore<TextToolOptions>({ ...DEFAULT_TEXT_OPTIONS }, textOptionsEqual),
  thumbnailVersion: createKeyedVersionStore(),
  thumbnailStatus: createKeyedStatusStore<LayerThumbnailStatus>(),
  adjustmentStatus: createKeyedStatusStore<LayerAdjustmentStatus>(),
  transformOptions: createScalarStore<TransformOptions>({
    smoothing: false,
    resampleMode: 'bicubic',
    fitMode: 'contain',
  }),
  transformSession: createScalarStore<TransformSession | null>(null, transformSessionEqual),
  pendingTransformLayerId: createScalarStore<string | null>(null),
  viewportReady: createScalarStore<boolean>(false),
  zoom: createScalarStore<number>(1),
});
