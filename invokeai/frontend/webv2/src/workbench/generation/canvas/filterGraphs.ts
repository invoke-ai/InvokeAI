/**
 * Pure control-filter graph builders (legacy parity).
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
 * builds the exact backend nodes used by legacy, fed `image: { image_name }`
 * plus the filter's parameters. The registry is the spec — settings editors
 * read `buildDefaults()` and the param metadata.
 */

import type { BackendGraphContract } from '@workbench/types';

/** The deterministic node id every single-node filter graph uses. */
export const FILTER_NODE_ID = 'control_filter';

/** A filter parameter's UI + validation metadata (drives the settings editor). */
export type FilterParamSpec =
  | {
      kind: 'number';
      key: string;
      default: number;
      min: number;
      max: number;
      step: number;
      integer?: boolean;
      sliderMin?: number;
      sliderMax?: number;
      coerceMin?: number;
      dynamicBounds?: 'adjust_value';
    }
  | { kind: 'boolean'; key: string; default: boolean }
  | {
      kind: 'enum';
      key: string;
      default: string;
      options: readonly { labelKey: string; value: string }[];
    }
  | { kind: 'string'; key: string; default: string }
  | { kind: 'model'; key: string; default: null; modelType: 'spandrel_image_to_image' };

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
      { default: 100, integer: true, key: 'low_threshold', kind: 'number', max: 255, min: 0, step: 1 },
      { default: 200, integer: true, key: 'high_threshold', kind: 'number', max: 255, min: 0, step: 1 },
    ],
    type: 'canny_edge_detection',
  },
  {
    params: [
      {
        default: 'small_v2',
        key: 'model_size',
        kind: 'enum',
        options: [
          { labelKey: 'widgets.layers.control.filterOptions.model_size.large', value: 'large' },
          { labelKey: 'widgets.layers.control.filterOptions.model_size.base', value: 'base' },
          { labelKey: 'widgets.layers.control.filterOptions.model_size.small', value: 'small' },
          { labelKey: 'widgets.layers.control.filterOptions.model_size.small_v2', value: 'small_v2' },
        ],
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
      { default: 0.1, key: 'score_threshold', kind: 'number', max: 1, min: 0, step: 0.01 },
      { default: 20, key: 'distance_threshold', kind: 'number', max: 1000, min: 0, sliderMax: 100, step: 1 },
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
    params: [{ default: 256, integer: true, key: 'scale_factor', kind: 'number', max: 4096, min: 0, step: 1 }],
    type: 'content_shuffle',
  },
  {
    params: [
      { default: 1, integer: true, key: 'max_faces', kind: 'number', max: 20, min: 1, step: 1 },
      { default: 0.5, key: 'min_confidence', kind: 'number', max: 1, min: 0, step: 0.01 },
    ],
    type: 'mediapipe_face_detection',
  },
  {
    params: [
      { default: 64, integer: true, key: 'tile_size', kind: 'number', max: 4096, min: 1, sliderMax: 256, step: 1 },
    ],
    type: 'color_map',
  },
  {
    params: [
      {
        default: 'Luminosity (LAB)',
        key: 'channel',
        kind: 'enum',
        options: [
          { labelKey: 'widgets.layers.control.filterOptions.channel.red', value: 'Red (RGBA)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.green', value: 'Green (RGBA)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.blue', value: 'Blue (RGBA)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.alpha', value: 'Alpha (RGBA)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.cyan', value: 'Cyan (CMYK)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.magenta', value: 'Magenta (CMYK)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.yellow', value: 'Yellow (CMYK)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.black', value: 'Black (CMYK)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.hue', value: 'Hue (HSV)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.saturation', value: 'Saturation (HSV)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.value', value: 'Value (HSV)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.luminosity', value: 'Luminosity (LAB)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.lab_a', value: 'A (LAB)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.lab_b', value: 'B (LAB)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.y', value: 'Y (YCbCr)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.cb', value: 'Cb (YCbCr)' },
          { labelKey: 'widgets.layers.control.filterOptions.channel.cr', value: 'Cr (YCbCr)' },
        ],
      },
      { default: 1, dynamicBounds: 'adjust_value', key: 'value', kind: 'number', max: 255, min: 0, step: 0.0025 },
      { default: false, key: 'scale_values', kind: 'boolean' },
    ],
    type: 'adjust_image',
  },
  { params: [], type: 'lineart_anime_edge_detection' },
  { params: [], type: 'normal_map' },
  {
    params: [
      { default: null, key: 'model', kind: 'model', modelType: 'spandrel_image_to_image' },
      { default: true, key: 'autoScale', kind: 'boolean' },
      { default: 1, key: 'scale', kind: 'number', max: 16, min: 1, step: 1 },
    ],
    type: 'spandrel_filter',
  },
  {
    params: [
      {
        default: 'gaussian',
        key: 'blur_type',
        kind: 'enum',
        options: [
          { labelKey: 'widgets.layers.control.filterOptions.blur_type.gaussian', value: 'gaussian' },
          { labelKey: 'widgets.layers.control.filterOptions.blur_type.box', value: 'box' },
        ],
      },
      { coerceMin: 0, default: 8, key: 'radius', kind: 'number', max: 4096, min: 1, sliderMax: 64, step: 0.1 },
    ],
    type: 'img_blur',
  },
  {
    params: [
      {
        default: 'gaussian',
        key: 'noise_type',
        kind: 'enum',
        options: [
          { labelKey: 'widgets.layers.control.filterOptions.noise_type.gaussian', value: 'gaussian' },
          {
            labelKey: 'widgets.layers.control.filterOptions.noise_type.salt_and_pepper',
            value: 'salt_and_pepper',
          },
        ],
      },
      { default: 0.3, key: 'amount', kind: 'number', max: 1, min: 0, step: 0.01 },
      { default: true, key: 'noise_color', kind: 'boolean' },
      { default: 1, integer: true, key: 'size', kind: 'number', max: 256, min: 1, sliderMax: 16, step: 1 },
    ],
    type: 'img_noise',
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

export interface FilterNumberBounds {
  inputMax: number;
  inputMin: number;
  sliderMax: number;
  sliderMin: number;
  step: number;
}

/** Resolves legacy slider/input ranges, including adjust_image's mode-dependent value maximum. */
export const getFilterNumberBounds = (
  param: Extract<FilterParamSpec, { kind: 'number' }>,
  settings: Record<string, unknown>
): FilterNumberBounds => {
  const dynamicMax = param.dynamicBounds === 'adjust_value' && settings.scale_values !== true ? 2 : param.max;
  return {
    inputMax: dynamicMax,
    inputMin: param.min,
    sliderMax: param.dynamicBounds === 'adjust_value' ? dynamicMax : (param.sliderMax ?? dynamicMax),
    sliderMin: param.sliderMin ?? param.min,
    step: param.step,
  };
};

const isNonemptyStringField = (value: object, key: string): boolean =>
  key in value &&
  typeof (value as Record<string, unknown>)[key] === 'string' &&
  ((value as Record<string, unknown>)[key] as string).trim().length > 0;

export const isSpandrelModelIdentifier = (value: unknown): boolean =>
  typeof value === 'object' &&
  value !== null &&
  ['key', 'hash', 'name', 'base', 'type'].every((key) => isNonemptyStringField(value, key)) &&
  (value as { type: string }).type === 'spandrel_image_to_image';

/** True when settings contain everything required to run the selected filter. */
export const isFilterConfigValid = (type: string, settings: Record<string, unknown>): boolean => {
  if (type !== 'spandrel_filter') {
    return FILTERS_BY_TYPE.has(type);
  }
  return isSpandrelModelIdentifier(settings.model);
};

/** Coerces arbitrary stored settings to the filter's params, falling back to each default. */
const resolveSettings = (
  definition: FilterDefinition,
  settings: Record<string, unknown> | undefined
): Record<string, unknown> => {
  const resolved: Record<string, unknown> = {};
  for (const param of definition.params) {
    const value = settings?.[param.key];
    if (param.kind === 'number' && typeof value === 'number' && Number.isFinite(value)) {
      const bounds = getFilterNumberBounds(param, settings ?? {});
      const clamped = Math.min(bounds.inputMax, Math.max(param.coerceMin ?? bounds.inputMin, value));
      resolved[param.key] = param.integer ? Math.round(clamped) : clamped;
    } else if (param.kind === 'boolean' && typeof value === 'boolean') {
      resolved[param.key] = value;
    } else if (
      param.kind === 'enum' &&
      typeof value === 'string' &&
      param.options.some((option) => option.value === value)
    ) {
      resolved[param.key] = value;
    } else if (param.kind === 'string' && typeof value === 'string') {
      resolved[param.key] = value;
    } else if (
      param.kind === 'model' &&
      param.modelType === 'spandrel_image_to_image' &&
      isSpandrelModelIdentifier(value)
    ) {
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

  const resolved = resolveSettings(definition, settings);
  const node = {
    id: FILTER_NODE_ID,
    image: { image_name: imageName },
    is_intermediate: true,
    type: definition.type,
    use_cache: true,
    ...resolved,
  };

  const graph: BackendGraphContract = {
    edges: [],
    id: `control-filter-${definition.type}`,
    nodes: { [FILTER_NODE_ID]: node },
  };
  const edge = (sourceId: string, sourceField: string, destinationId: string, destinationField: string): void => {
    graph.edges.push({
      destination: { field: destinationField, node_id: destinationId },
      source: { field: sourceField, node_id: sourceId },
    });
  };

  if (filterType === 'adjust_image') {
    const { channel, scale_values: scaleValues, value } = resolved;
    graph.nodes[FILTER_NODE_ID] = scaleValues
      ? {
          ...node,
          channel,
          invert_channel: false,
          scale: value,
          scale_values: undefined,
          type: 'img_channel_multiply',
          value: undefined,
        }
      : {
          ...node,
          channel,
          offset: Math.round(255 * ((value as number) - 1)),
          scale_values: undefined,
          type: 'img_channel_offset',
          value: undefined,
        };
  } else if (filterType === 'spandrel_filter') {
    const model = resolved.model;
    if (!model) {
      throw new Error('Spandrel filter requires an image-to-image model.');
    }
    graph.nodes[FILTER_NODE_ID] = resolved.autoScale
      ? {
          ...node,
          autoScale: undefined,
          image_to_image_model: model,
          model: undefined,
          type: 'spandrel_image_to_image_autoscale',
        }
      : {
          ...node,
          autoScale: undefined,
          image_to_image_model: model,
          model: undefined,
          scale: undefined,
          type: 'spandrel_image_to_image',
        };
  } else if (filterType === 'img_blur') {
    const radius = resolved.radius as number;
    const padding = Math.max(0, Math.ceil(radius * (resolved.blur_type === 'gaussian' ? 3 : 1)));
    if (padding > 0) {
      delete graph.nodes[FILTER_NODE_ID]!.image;
      graph.nodes.control_filter_pad = {
        bottom: padding,
        id: 'control_filter_pad',
        image: { image_name: imageName },
        left: padding,
        right: padding,
        top: padding,
        type: 'img_pad_crop',
        use_cache: true,
      };
      graph.nodes.control_filter_alpha = {
        channel: 'Alpha (RGBA)',
        id: 'control_filter_alpha',
        is_intermediate: true,
        offset: 1,
        type: 'img_channel_offset',
        use_cache: true,
      };
      edge('control_filter_pad', 'image', FILTER_NODE_ID, 'image');
      edge(FILTER_NODE_ID, 'image', 'control_filter_alpha', 'image');
      return { graph, outputNodeId: 'control_filter_alpha' };
    }
  } else if (filterType === 'img_noise') {
    graph.nodes.control_filter_seed = {
      high: 2147483647,
      id: 'control_filter_seed',
      low: 0,
      type: 'rand_int',
      use_cache: false,
    };
    edge('control_filter_seed', 'value', FILTER_NODE_ID, 'seed');
  }

  return {
    graph,
    outputNodeId: FILTER_NODE_ID,
  };
};
