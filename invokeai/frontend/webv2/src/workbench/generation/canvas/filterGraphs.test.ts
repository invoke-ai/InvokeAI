import { describe, expect, it } from 'vitest';

import {
  buildFilterDefaults,
  buildFilterGraph,
  CONTROL_FILTERS,
  DEFAULT_CONTROL_FILTER_TYPE,
  FILTER_NODE_ID,
  getFilterDefinition,
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
};

const EXPECTED_TYPES = Object.keys(EXPECTED_DEFAULTS);

describe('CONTROL_FILTERS registry', () => {
  it('contains exactly the ten legacy filter types', () => {
    const types = CONTROL_FILTERS.map((definition) => definition.type);
    expect(types).toHaveLength(10);
    expect(new Set(types).size).toBe(10);
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

  it('builds a single-node graph with default params for every filter', () => {
    for (const type of EXPECTED_TYPES) {
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
