import { describe, expect, it } from 'vitest';

import { buildFilterGraph, CONTROL_FILTERS, FILTER_NODE_ID } from './filterGraphs';

/** The exact legacy defaults for each supported filter — the spec under test. */
const EXPECTED_DEFAULTS: Record<string, Record<string, unknown>> = {
  canny_edge_detection: { low_threshold: 100, high_threshold: 200 },
  depth_anything_depth_estimation: { model_size: 'small_v2' },
  dw_openpose_detection: { draw_body: true, draw_face: true, draw_hands: true },
  lineart_edge_detection: { coarse: false },
  hed_edge_detection: { scribble: false },
  mlsd_detection: { score_threshold: 0.1, distance_threshold: 20 },
  pidi_edge_detection: { quantize_edges: false, scribble: false },
  scribble_edge_detection: {},
  content_shuffle: { scale_factor: 256 },
  mediapipe_face_detection: { max_faces: 1, min_confidence: 0.5 },
  color_map: { tile_size: 64 },
  adjust_image: { channel: 'Luminosity (LAB)', value: 1, scale_values: false },
  lineart_anime_edge_detection: {},
  normal_map: {},
  spandrel_filter: { model: null, autoScale: true, scale: 1 },
  img_blur: { blur_type: 'gaussian', radius: 8 },
  img_noise: { noise_type: 'gaussian', amount: 0.3, noise_color: true, size: 1 },
  grayscale: {},
};

const EXPECTED_TYPES = Object.keys(EXPECTED_DEFAULTS);
const SPANDREL_MODEL = {
  base: 'any',
  hash: 'blake3-hash',
  key: 'upscale',
  name: 'Upscaler',
  type: 'spandrel_image_to_image',
};

describe('buildFilterGraph', () => {
  const imageName = 'source-image.png';

  it('builds the direct legacy filter nodes with default params', () => {
    const directTypes = EXPECTED_TYPES.filter(
      (type) =>
        !['adjust_image', 'spandrel_filter', 'img_blur', 'img_noise', 'scribble_edge_detection', 'grayscale'].includes(
          type
        )
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
      // Previews run intermediate (out of the gallery, GC-eligible); only Apply promotes the chosen image to durable
      // via makeImageDurable.
      expect(node.is_intermediate).toBe(true);
      expect(node.use_cache).toBe(true);

      // Every default param field is present on the node.
      for (const [key, value] of Object.entries(EXPECTED_DEFAULTS[type])) {
        expect(node[key]).toBe(value);
      }
    }
  });

  it('builds Grayscale as a mode round trip, so the layer stays RGB', () => {
    const { graph, outputNodeId } = buildFilterGraph('grayscale', imageName);

    expect(graph.nodes[FILTER_NODE_ID]).toMatchObject({
      image: { image_name: imageName },
      mode: 'L',
      type: 'img_conv',
    });
    expect(graph.nodes.control_filter_rgb).toEqual({
      id: 'control_filter_rgb',
      is_intermediate: true,
      mode: 'RGB',
      type: 'img_conv',
      use_cache: true,
    });
    expect(graph.edges).toEqual([
      {
        destination: { field: 'image', node_id: 'control_filter_rgb' },
        source: { field: 'image', node_id: FILTER_NODE_ID },
      },
    ]);
    expect(outputNodeId).toBe('control_filter_rgb');
  });

  it('flattens transparent control inputs over white before processing and reapplies their alpha', () => {
    const { graph, outputNodeId } = buildFilterGraph('canny_edge_detection', imageName, undefined, {
      height: 80,
      preserveTransparency: true,
      width: 120,
    });

    expect(graph.nodes.control_filter_background).toMatchObject({
      color: { a: 255, b: 255, g: 255, r: 255 },
      height: 80,
      type: 'blank_image',
      width: 120,
    });
    expect(graph.nodes.control_filter_flatten).toMatchObject({
      crop: true,
      image: { image_name: imageName },
      type: 'img_paste',
    });
    expect(graph.nodes.control_filter_source_alpha).toMatchObject({
      image: { image_name: imageName },
      type: 'tomask',
    });
    expect(graph.nodes.control_filter_restore_alpha).toMatchObject({
      is_intermediate: true,
      type: 'apply_mask_to_image',
    });
    expect(graph.nodes[FILTER_NODE_ID]).not.toHaveProperty('image');
    expect(graph.edges).toEqual(
      expect.arrayContaining([
        {
          destination: { field: 'image', node_id: FILTER_NODE_ID },
          source: { field: 'image', node_id: 'control_filter_flatten' },
        },
        {
          destination: { field: 'image', node_id: 'control_filter_restore_alpha' },
          source: { field: 'image', node_id: FILTER_NODE_ID },
        },
        {
          destination: { field: 'mask', node_id: 'control_filter_restore_alpha' },
          source: { field: 'image', node_id: 'control_filter_source_alpha' },
        },
      ])
    );
    expect(outputNodeId).toBe('control_filter_restore_alpha');
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
});
