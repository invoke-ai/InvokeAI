/**
 * Pure single-node control-filter graph builders (legacy parity).
 *
 * A control layer can non-destructively preview a filter (canny, depth, pose, …)
 * over its composited source. This module names each supported filter, its
 * legacy-default settings, and the one-node backend graph it compiles to — with
 * NO fetch, NO engine, NO React. The executor composites + uploads the layer's
 * source image, then {@link buildFilterGraph} shapes the graph the utility queue
 * runs; its `outputNodeId` names the node whose output image is the preview.
 *
 * Node type strings + defaults mirror legacy
 * `features/controlLayers/store/filters.ts` (`IMAGE_FILTERS`): every filter here
 * builds a single node whose backend `type` equals the filter id, fed
 * `image: { image_name }` plus the filter's parameters. The registry is the
 * spec — settings editors read `buildDefaults()` and the param metadata.
 */

import type { BackendGraphContract } from '@workbench/types';

/** The deterministic node id every single-node filter graph uses. */
export const FILTER_NODE_ID = 'control_filter';

/** A filter parameter's UI + validation metadata (drives the settings editor). */
export type FilterParamSpec =
  | { kind: 'number'; key: string; default: number; min?: number; max?: number; step?: number; integer?: boolean }
  | { kind: 'boolean'; key: string; default: boolean }
  | { kind: 'enum'; key: string; default: string; options: readonly string[] };

/** One filter's identity, defaults, and node-arg projection. */
export interface FilterDefinition {
  /** The filter id (equals the backend node `type`). */
  type: string;
  /** Ordered parameter specs (also the default-settings source). */
  params: readonly FilterParamSpec[];
}

/** Builds the default settings object for a filter definition. */
export const buildFilterDefaults = (definition: FilterDefinition): Record<string, unknown> => {
  const settings: Record<string, unknown> = {};
  for (const param of definition.params) {
    settings[param.key] = param.default;
  }
  return settings;
};

/**
 * The supported filters, in the legacy display order of the required set
 * (`features/controlLayers/store/filters.ts`). Each id equals its backend node
 * `type`; params carry legacy defaults + ranges.
 */
export const CONTROL_FILTERS: readonly FilterDefinition[] = [
  {
    params: [
      { default: 100, integer: true, key: 'low_threshold', kind: 'number', max: 255, min: 0 },
      { default: 200, integer: true, key: 'high_threshold', kind: 'number', max: 255, min: 0 },
    ],
    type: 'canny_edge_detection',
  },
  {
    params: [
      {
        default: 'small_v2',
        key: 'model_size',
        kind: 'enum',
        options: ['large', 'base', 'small', 'small_v2'],
      },
    ],
    type: 'depth_anything_depth_estimation',
  },
  {
    params: [
      { default: true, key: 'draw_body', kind: 'boolean' },
      { default: true, key: 'draw_face', kind: 'boolean' },
      { default: true, key: 'draw_hands', kind: 'boolean' },
    ],
    type: 'dw_openpose_detection',
  },
  {
    params: [{ default: false, key: 'coarse', kind: 'boolean' }],
    type: 'lineart_edge_detection',
  },
  {
    params: [{ default: false, key: 'scribble', kind: 'boolean' }],
    type: 'hed_edge_detection',
  },
  {
    params: [
      { default: 0.1, key: 'score_threshold', kind: 'number', min: 0, step: 0.01 },
      { default: 20, key: 'distance_threshold', kind: 'number', min: 0, step: 0.1 },
    ],
    type: 'mlsd_detection',
  },
  {
    params: [
      { default: false, key: 'quantize_edges', kind: 'boolean' },
      { default: false, key: 'scribble', kind: 'boolean' },
    ],
    type: 'pidi_edge_detection',
  },
  {
    params: [{ default: 256, integer: true, key: 'scale_factor', kind: 'number', min: 1 }],
    type: 'content_shuffle',
  },
  {
    params: [
      { default: 1, integer: true, key: 'max_faces', kind: 'number', min: 1 },
      { default: 0.5, key: 'min_confidence', kind: 'number', max: 1, min: 0, step: 0.01 },
    ],
    type: 'mediapipe_face_detection',
  },
  {
    params: [{ default: 64, integer: true, key: 'tile_size', kind: 'number', min: 1 }],
    type: 'color_map',
  },
];

const FILTERS_BY_TYPE: ReadonlyMap<string, FilterDefinition> = new Map(
  CONTROL_FILTERS.map((definition) => [definition.type, definition])
);

/** The default filter type applied when a control layer's filter section is enabled. */
export const DEFAULT_CONTROL_FILTER_TYPE = 'canny_edge_detection';

/** Looks up a filter definition by its type id (`undefined` for unknown filters). */
export const getFilterDefinition = (type: string): FilterDefinition | undefined => FILTERS_BY_TYPE.get(type);

/** True when `type` names a supported control filter. */
export const isSupportedFilterType = (type: string): boolean => FILTERS_BY_TYPE.has(type);

/** Coerces arbitrary stored settings to the filter's params, falling back to each default. */
const resolveSettings = (
  definition: FilterDefinition,
  settings: Record<string, unknown> | undefined
): Record<string, unknown> => {
  const resolved: Record<string, unknown> = {};
  for (const param of definition.params) {
    const value = settings?.[param.key];
    if (param.kind === 'number' && typeof value === 'number' && Number.isFinite(value)) {
      resolved[param.key] = param.integer ? Math.round(value) : value;
    } else if (param.kind === 'boolean' && typeof value === 'boolean') {
      resolved[param.key] = value;
    } else if (param.kind === 'enum' && typeof value === 'string' && param.options.includes(value)) {
      resolved[param.key] = value;
    } else {
      resolved[param.key] = param.default;
    }
  }
  return resolved;
};

/** The result of {@link buildFilterGraph}: a one-node graph + the output node id. */
export interface FilterGraphResult {
  graph: BackendGraphContract;
  outputNodeId: string;
}

/**
 * Builds the single-node filter graph for `filterType` over `imageName`. Throws
 * for an unknown filter type. Unknown / out-of-range settings fall back to the
 * filter's legacy defaults, so a malformed persisted `filter.settings` still
 * produces a valid graph. The output image is INTERMEDIATE (`is_intermediate:
 * true`): every preview click runs this graph, and an intermediate output stays
 * out of the gallery and is garbage-collected. Only on "Apply" does the caller
 * promote the chosen image to durable (`makeImageDurable`), so it survives as
 * the layer's new source without every preview littering the gallery.
 */
export const buildFilterGraph = (
  filterType: string,
  imageName: string,
  settings?: Record<string, unknown>
): FilterGraphResult => {
  const definition = FILTERS_BY_TYPE.get(filterType);
  if (!definition) {
    throw new Error(`Unknown control filter "${filterType}".`);
  }

  const node = {
    id: FILTER_NODE_ID,
    image: { image_name: imageName },
    is_intermediate: true,
    type: definition.type,
    use_cache: true,
    ...resolveSettings(definition, settings),
  };

  return {
    graph: {
      edges: [],
      id: `control-filter-${definition.type}`,
      nodes: { [FILTER_NODE_ID]: node },
    },
    outputNodeId: FILTER_NODE_ID,
  };
};
