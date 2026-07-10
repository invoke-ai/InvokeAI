import type { BackendGraphEdgeContract } from '@workbench/types';

import { describe, expect, it } from 'vitest';

import { buildSamGraph } from './samGraph';

const edge = (
  sourceNodeId: string,
  sourceField: string,
  targetNodeId: string,
  targetField: string
): BackendGraphEdgeContract => ({
  destination: { field: targetField, node_id: targetNodeId },
  source: { field: sourceField, node_id: sourceNodeId },
});

describe('buildSamGraph', () => {
  it.each([
    ['segment-anything-2-large', false],
    ['segment-anything-huge', true],
  ] as const)('builds the exact prompt-to-mask graph for %s with refinement=%s', (model, applyPolygonRefinement) => {
    const { graph, outputNodeId } = buildSamGraph({
      applyPolygonRefinement,
      imageName: 'source.png',
      model,
      prompt: 'cat',
    });

    expect(outputNodeId).toBe('sam-output');
    expect(graph).toEqual({
      edges: [
        edge('sam-source', 'image', 'sam-detect', 'image'),
        edge('sam-source', 'image', 'sam-segment', 'image'),
        edge('sam-detect', 'collection', 'sam-segment', 'bounding_boxes'),
        edge('sam-segment', 'mask', 'sam-output', 'mask'),
      ],
      id: 'select-object-sam',
      nodes: {
        'sam-detect': {
          detection_threshold: 0.3,
          id: 'sam-detect',
          is_intermediate: true,
          model: 'grounding-dino-base',
          prompt: 'cat',
          type: 'grounding_dino',
          use_cache: true,
        },
        'sam-output': {
          id: 'sam-output',
          image: { image_name: 'source.png' },
          invert: false,
          is_intermediate: true,
          type: 'apply_tensor_mask_to_image',
          use_cache: true,
        },
        'sam-segment': {
          apply_polygon_refinement: applyPolygonRefinement,
          id: 'sam-segment',
          is_intermediate: true,
          mask_filter: 'largest',
          model,
          type: 'segment_anything',
          use_cache: true,
        },
        'sam-source': {
          id: 'sam-source',
          image: { image_name: 'source.png' },
          is_intermediate: true,
          type: 'image',
          use_cache: true,
        },
      },
    });
  });

  it('rejects a blank object prompt', () => {
    expect(() =>
      buildSamGraph({
        applyPolygonRefinement: false,
        imageName: 'source.png',
        model: 'segment-anything-2-large',
        prompt: '   ',
      })
    ).toThrow('A Segment Anything object prompt is required.');
  });
});
