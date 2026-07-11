import { describe, expect, it } from 'vitest';

import {
  buildFilterDefaults,
  buildFilterGraph,
  CONTROL_FILTERS,
  DEFAULT_CONTROL_FILTER_TYPE,
  FILTER_NODE_ID,
  getFilterNumberBounds,
  getFilterDefinition,
  isFilterConfigValid,
  isSupportedFilterType,
} from './filterGraphs';

/** The exact legacy defaults for each supported filter — the spec under test. */
const EXPECTED_DEFAULTS: Record<string, Record<string, unknown>> = {
  canny_edge_detection: { low_threshold: 100, high_threshold: 200 },
  depth_anything_depth_estimation: { model_size: 'small_v2' },
  dw_openpose_detection: { draw_body: true, draw_face: true, draw_hands: true },
  lineart_edge_detection: { coarse: false },
  hed_edge_detection: { scribble: false },
  mlsd_detection: { score_threshold: 0.1, distance_threshold: 20 },
  pidi_edge_detection: { quantize_edges: false, scribble: false },
  content_shuffle: { scale_factor: 256 },
  mediapipe_face_detection: { max_faces: 1, min_confidence: 0.5 },
  color_map: { tile_size: 64 },
  adjust_image: { channel: 'Luminosity (LAB)', value: 1, scale_values: false },
  lineart_anime_edge_detection: {},
  normal_map: {},
  spandrel_filter: { model: null, autoScale: true, scale: 1 },
  img_blur: { blur_type: 'gaussian', radius: 8 },
  img_noise: { noise_type: 'gaussian', amount: 0.3, noise_color: true, size: 1 },
};

const EXPECTED_TYPES = Object.keys(EXPECTED_DEFAULTS);
const SPANDREL_MODEL = {
  base: 'any',
  hash: 'blake3-hash',
  key: 'upscale',
  name: 'Upscaler',
  type: 'spandrel_image_to_image',
};

describe('CONTROL_FILTERS registry', () => {
  it('contains exactly all sixteen legacy filter types', () => {
    const types = CONTROL_FILTERS.map((definition) => definition.type);
    expect(types).toHaveLength(16);
    expect(new Set(types).size).toBe(16);
    expect(types).toEqual(EXPECTED_TYPES);
  });

  it('projects each definition to its exact legacy defaults', () => {
    for (const definition of CONTROL_FILTERS) {
      expect(buildFilterDefaults(definition)).toEqual(EXPECTED_DEFAULTS[definition.type]);
    }
  });

  it('names canny edge detection as the default control filter', () => {
    expect(DEFAULT_CONTROL_FILTER_TYPE).toBe('canny_edge_detection');
  });

  it('declares localized labels for every enum option', () => {
    for (const definition of CONTROL_FILTERS) {
      for (const param of definition.params) {
        if (param.kind === 'enum') {
          expect(
            param.options.every((option) => option.labelKey.startsWith('widgets.layers.control.filterOptions.'))
          ).toBe(true);
        }
      }
    }
  });
});

describe('getFilterNumberBounds', () => {
  it.each([
    [
      'canny_edge_detection',
      'low_threshold',
      {},
      { inputMax: 255, inputMin: 0, sliderMax: 255, sliderMin: 0, step: 1 },
    ],
    [
      'canny_edge_detection',
      'high_threshold',
      {},
      { inputMax: 255, inputMin: 0, sliderMax: 255, sliderMin: 0, step: 1 },
    ],
    ['mlsd_detection', 'score_threshold', {}, { inputMax: 1, inputMin: 0, sliderMax: 1, sliderMin: 0, step: 0.01 }],
    ['content_shuffle', 'scale_factor', {}, { inputMax: 4096, inputMin: 0, sliderMax: 4096, sliderMin: 0, step: 1 }],
    ['mediapipe_face_detection', 'max_faces', {}, { inputMax: 20, inputMin: 1, sliderMax: 20, sliderMin: 1, step: 1 }],
    [
      'mediapipe_face_detection',
      'min_confidence',
      {},
      { inputMax: 1, inputMin: 0, sliderMax: 1, sliderMin: 0, step: 0.01 },
    ],
    ['color_map', 'tile_size', {}, { inputMax: 4096, inputMin: 1, sliderMax: 256, sliderMin: 1, step: 1 }],
    [
      'mlsd_detection',
      'distance_threshold',
      {},
      { inputMax: 1000, inputMin: 0, sliderMax: 100, sliderMin: 0, step: 1 },
    ],
    ['spandrel_filter', 'scale', {}, { inputMax: 16, inputMin: 1, sliderMax: 16, sliderMin: 1, step: 1 }],
    ['img_blur', 'radius', {}, { inputMax: 4096, inputMin: 1, sliderMax: 64, sliderMin: 1, step: 0.1 }],
    ['img_noise', 'amount', {}, { inputMax: 1, inputMin: 0, sliderMax: 1, sliderMin: 0, step: 0.01 }],
    ['img_noise', 'size', {}, { inputMax: 256, inputMin: 1, sliderMax: 16, sliderMin: 1, step: 1 }],
  ])('returns legacy slider and numeric bounds for %s.%s', (type, key, settings, expected) => {
    const param = getFilterDefinition(type)?.params.find((candidate) => candidate.key === key);
    if (param?.kind !== 'number') {
      throw new Error(`Missing number param ${type}.${key}`);
    }
    expect(getFilterNumberBounds(param, settings)).toEqual(expected);
  });

  it('changes adjust value bounds with scale_values', () => {
    const param = getFilterDefinition('adjust_image')?.params.find((candidate) => candidate.key === 'value');
    if (param?.kind !== 'number') {
      throw new Error('Missing adjust_image.value');
    }
    expect(getFilterNumberBounds(param, { scale_values: false })).toMatchObject({ inputMax: 2, sliderMax: 2 });
    expect(getFilterNumberBounds(param, { scale_values: true })).toMatchObject({ inputMax: 255, sliderMax: 255 });
  });
});

describe('isSupportedFilterType / getFilterDefinition', () => {
  it('recognises every supported filter', () => {
    for (const type of EXPECTED_TYPES) {
      expect(isSupportedFilterType(type)).toBe(true);
      expect(getFilterDefinition(type)?.type).toBe(type);
    }
  });

  it('rejects an unknown filter type', () => {
    expect(isSupportedFilterType('not_a_filter')).toBe(false);
    expect(getFilterDefinition('not_a_filter')).toBeUndefined();
  });
});

describe('buildFilterGraph', () => {
  const imageName = 'source-image.png';

  it('builds the direct legacy filter nodes with default params', () => {
    const directTypes = EXPECTED_TYPES.filter(
      (type) => !['adjust_image', 'spandrel_filter', 'img_blur', 'img_noise'].includes(type)
    );
    for (const type of directTypes) {
      const { graph, outputNodeId } = buildFilterGraph(type, imageName);

      expect(outputNodeId).toBe(FILTER_NODE_ID);
      expect(graph.edges).toEqual([]);
      expect(typeof graph.id).toBe('string');
      expect(graph.id).toContain(type);
      expect(Object.keys(graph.nodes)).toEqual([FILTER_NODE_ID]);

      const node = graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
      expect(node.id).toBe(FILTER_NODE_ID);
      expect(node.type).toBe(type);
      expect(node.image).toEqual({ image_name: imageName });
      // Previews run intermediate (out of the gallery, GC-eligible); only Apply
      // promotes the chosen image to durable via makeImageDurable.
      expect(node.is_intermediate).toBe(true);
      expect(node.use_cache).toBe(true);

      // Every default param field is present on the node.
      for (const [key, value] of Object.entries(EXPECTED_DEFAULTS[type])) {
        expect(node[key]).toBe(value);
      }
    }
  });

  it('builds adjust_image as offset or channel multiply', () => {
    const offset = buildFilterGraph('adjust_image', imageName).graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(offset).toMatchObject({
      channel: 'Luminosity (LAB)',
      image: { image_name: imageName },
      offset: 0,
      type: 'img_channel_offset',
    });

    const multiplied = buildFilterGraph('adjust_image', imageName, {
      channel: 'Red (RGBA)',
      scale_values: true,
      value: 1.5,
    }).graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(multiplied).toMatchObject({
      channel: 'Red (RGBA)',
      image: { image_name: imageName },
      invert_channel: false,
      scale: 1.5,
      type: 'img_channel_multiply',
    });

    const multipliedAtLegacyMax = buildFilterGraph('adjust_image', imageName, {
      scale_values: true,
      value: 999,
    }).graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(multipliedAtLegacyMax.scale).toBe(255);

    const offsetAtMax = buildFilterGraph('adjust_image', imageName, {
      scale_values: false,
      value: 255,
    }).graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(offsetAtMax.offset).toBe(255);
  });

  it('builds padded gaussian blur with alpha nudge wiring', () => {
    const { graph, outputNodeId } = buildFilterGraph('img_blur', imageName);
    expect(graph.nodes.control_filter_pad).toMatchObject({
      bottom: 24,
      image: { image_name: imageName },
      left: 24,
      right: 24,
      top: 24,
      type: 'img_pad_crop',
    });
    expect(graph.nodes[FILTER_NODE_ID]).toMatchObject({ blur_type: 'gaussian', radius: 8, type: 'img_blur' });
    expect(graph.nodes.control_filter_alpha).toMatchObject({
      channel: 'Alpha (RGBA)',
      offset: 1,
      type: 'img_channel_offset',
    });
    expect(graph.edges).toEqual([
      {
        destination: { field: 'image', node_id: FILTER_NODE_ID },
        source: { field: 'image', node_id: 'control_filter_pad' },
      },
      {
        destination: { field: 'image', node_id: 'control_filter_alpha' },
        source: { field: 'image', node_id: FILTER_NODE_ID },
      },
    ]);
    expect(outputNodeId).toBe('control_filter_alpha');
  });

  it('builds zero-radius blur directly without padding', () => {
    const { graph, outputNodeId } = buildFilterGraph('img_blur', imageName, { radius: 0 });
    expect(Object.keys(graph.nodes)).toEqual([FILTER_NODE_ID]);
    expect(graph.nodes[FILTER_NODE_ID]).toMatchObject({ image: { image_name: imageName }, radius: 0 });
    expect(graph.edges).toEqual([]);
    expect(outputNodeId).toBe(FILTER_NODE_ID);
  });

  it('builds noise with an uncached rand_int seed edge', () => {
    const { graph } = buildFilterGraph('img_noise', imageName);
    expect(graph.nodes[FILTER_NODE_ID]).toMatchObject({ amount: 0.3, noise_color: true, size: 1, type: 'img_noise' });
    expect(graph.nodes.control_filter_seed).toMatchObject({
      high: 2147483647,
      low: 0,
      type: 'rand_int',
      use_cache: false,
    });
    expect(graph.edges).toContainEqual({
      destination: { field: 'seed', node_id: FILTER_NODE_ID },
      source: { field: 'value', node_id: 'control_filter_seed' },
    });
  });

  it('builds Spandrel autoscale and direct nodes and rejects a missing model', () => {
    expect(() => buildFilterGraph('spandrel_filter', imageName)).toThrow(/model/i);
    expect(
      buildFilterGraph('spandrel_filter', imageName, { model: SPANDREL_MODEL, scale: 4 }).graph.nodes[FILTER_NODE_ID]
    ).toMatchObject({
      image_to_image_model: SPANDREL_MODEL,
      scale: 4,
      type: 'spandrel_image_to_image_autoscale',
    });
    const direct = buildFilterGraph('spandrel_filter', imageName, { autoScale: false, model: SPANDREL_MODEL }).graph
      .nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(direct).toMatchObject({ image_to_image_model: SPANDREL_MODEL, type: 'spandrel_image_to_image' });
    expect(direct.scale).toBeUndefined();
  });

  it('clamps and coerces persisted numeric settings to declared bounds', () => {
    const canny = buildFilterGraph('canny_edge_detection', imageName, {
      high_threshold: 999,
      low_threshold: -12.8,
    }).graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(canny.low_threshold).toBe(0);
    expect(canny.high_threshold).toBe(255);

    const noise = buildFilterGraph('img_noise', imageName, { amount: -2, size: 0.2 }).graph.nodes[
      FILTER_NODE_ID
    ] as Record<string, unknown>;
    expect(noise.amount).toBe(0);
    expect(noise.size).toBe(1);
  });

  it('coerces malformed persisted settings for every parameter of all sixteen filters', () => {
    for (const definition of CONTROL_FILTERS) {
      const malformed: Record<string, unknown> = Object.fromEntries(
        definition.params.map((param) => [param.key, param.kind === 'boolean' ? 'false' : { invalid: true }])
      );
      if (definition.type === 'spandrel_filter') {
        malformed.model = SPANDREL_MODEL;
      }
      const node = buildFilterGraph(definition.type, imageName, malformed).graph.nodes[FILTER_NODE_ID] as Record<
        string,
        unknown
      >;
      for (const param of definition.params) {
        if (param.kind === 'model') {
          continue;
        }
        const projectedKey = definition.type === 'adjust_image' && param.key === 'value' ? 'offset' : param.key;
        if (definition.type === 'adjust_image' && param.key === 'value') {
          expect(node[projectedKey], definition.type).toBe(0);
        } else if (
          (definition.type !== 'adjust_image' || param.key !== 'scale_values') &&
          (definition.type !== 'spandrel_filter' || param.key !== 'autoScale')
        ) {
          expect(node[projectedKey], `${definition.type}.${param.key}`).toBe(param.default);
        }
      }
    }
  });

  it('clamps every declared numeric parameter at both bounds', () => {
    for (const definition of CONTROL_FILTERS) {
      for (const param of definition.params) {
        if (param.kind !== 'number') {
          continue;
        }
        const lowSettings = { model: SPANDREL_MODEL, [param.key]: Number.MIN_SAFE_INTEGER };
        const highSettings = { model: SPANDREL_MODEL, [param.key]: Number.MAX_SAFE_INTEGER };
        const low = buildFilterGraph(definition.type, imageName, lowSettings).graph.nodes[FILTER_NODE_ID] as Record<
          string,
          unknown
        >;
        const high = buildFilterGraph(definition.type, imageName, highSettings).graph.nodes[FILTER_NODE_ID] as Record<
          string,
          unknown
        >;
        if (definition.type === 'adjust_image' && param.key === 'value') {
          expect(low.offset).toBe(-255);
          expect(high.offset).toBe(255);
        } else if (param.min !== undefined) {
          expect(low[param.key], `${definition.type}.${param.key}.min`).toBe(param.coerceMin ?? param.min);
        }
        if (param.max !== undefined && !(definition.type === 'adjust_image' && param.key === 'value')) {
          expect(high[param.key], `${definition.type}.${param.key}.max`).toBe(param.max);
        }
      }
    }
  });

  it('requires a complete nonempty backend Spandrel model identifier', () => {
    expect(isFilterConfigValid('spandrel_filter', { model: SPANDREL_MODEL })).toBe(true);
    for (const field of ['key', 'hash', 'name', 'base', 'type'] as const) {
      expect(isFilterConfigValid('spandrel_filter', { model: { ...SPANDREL_MODEL, [field]: '' } }), field).toBe(false);
    }
    expect(isFilterConfigValid('spandrel_filter', { model: { ...SPANDREL_MODEL, type: 'controlnet' } })).toBe(false);
    expect(() => buildFilterGraph('spandrel_filter', imageName, { model: { ...SPANDREL_MODEL, hash: '' } })).toThrow(
      /model/i
    );
  });

  it('applies valid custom settings, overriding defaults', () => {
    const { graph } = buildFilterGraph('canny_edge_detection', imageName, {
      low_threshold: 50,
      high_threshold: 220,
    });
    const node = graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(node.low_threshold).toBe(50);
    expect(node.high_threshold).toBe(220);
  });

  it('rounds non-integer values for integer params', () => {
    const { graph } = buildFilterGraph('canny_edge_detection', imageName, {
      low_threshold: 50.4,
      high_threshold: 199.6,
    });
    const node = graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(node.low_threshold).toBe(50);
    expect(node.high_threshold).toBe(200);
  });

  it('accepts a finite non-integer for a non-integer number param', () => {
    const { graph } = buildFilterGraph('mediapipe_face_detection', imageName, {
      min_confidence: 0.75,
    });
    const node = graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(node.min_confidence).toBe(0.75);
  });

  it('falls back to the default for wrong-typed number settings', () => {
    const badString = buildFilterGraph('canny_edge_detection', imageName, {
      low_threshold: 'bad',
    });
    expect((badString.graph.nodes[FILTER_NODE_ID] as Record<string, unknown>).low_threshold).toBe(100);

    const nan = buildFilterGraph('canny_edge_detection', imageName, {
      low_threshold: Number.NaN,
    });
    expect((nan.graph.nodes[FILTER_NODE_ID] as Record<string, unknown>).low_threshold).toBe(100);
  });

  it('falls back to the default for an unknown enum value', () => {
    const { graph } = buildFilterGraph('depth_anything_depth_estimation', imageName, {
      model_size: 'nonsense',
    });
    expect((graph.nodes[FILTER_NODE_ID] as Record<string, unknown>).model_size).toBe('small_v2');
  });

  it('accepts a valid enum value', () => {
    const { graph } = buildFilterGraph('depth_anything_depth_estimation', imageName, {
      model_size: 'large',
    });
    expect((graph.nodes[FILTER_NODE_ID] as Record<string, unknown>).model_size).toBe('large');
  });

  it('falls back to the default for a non-boolean boolean setting', () => {
    const { graph } = buildFilterGraph('dw_openpose_detection', imageName, {
      draw_body: 'yes',
      draw_face: 1,
    });
    const node = graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(node.draw_body).toBe(true);
    expect(node.draw_face).toBe(true);
    expect(node.draw_hands).toBe(true);
  });

  it('honours a valid boolean override', () => {
    const { graph } = buildFilterGraph('lineart_edge_detection', imageName, { coarse: true });
    expect((graph.nodes[FILTER_NODE_ID] as Record<string, unknown>).coarse).toBe(true);
  });

  it('ignores unknown setting keys', () => {
    const { graph } = buildFilterGraph('color_map', imageName, { unknown_key: 999 });
    const node = graph.nodes[FILTER_NODE_ID] as Record<string, unknown>;
    expect(node.tile_size).toBe(64);
    expect(node.unknown_key).toBeUndefined();
  });

  it('throws for an unknown filter type', () => {
    expect(() => buildFilterGraph('not_a_filter', imageName)).toThrow();
  });
});
