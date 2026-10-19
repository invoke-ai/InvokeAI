import type { BackendGraphEdgeContract } from '@workbench/graphContracts';

import { describe, expect, it } from 'vitest';

import { buildSamGraph, documentToExportLocalSamInput, isSamDocumentInputValid, isSamInputValid } from './samGraph';

const edge = (
  sourceNodeId: string,
  sourceField: string,
  targetNodeId: string,
  targetField: string
): BackendGraphEdgeContract => ({
  destination: { field: targetField, node_id: targetNodeId },
  source: { field: sourceField, node_id: sourceNodeId },
});

const baseNodes = {
  'sam-output': {
    id: 'sam-output',
    image: { image_name: 'source.png' },
    invert: false,
    is_intermediate: true,
    type: 'apply_tensor_mask_to_image',
    use_cache: true,
  },
  'sam-source': {
    id: 'sam-source',
    image: { image_name: 'source.png' },
    is_intermediate: true,
    type: 'image',
    use_cache: true,
  },
};

describe('buildSamGraph', () => {
  it('builds the exact prompt graph through Grounding DINO', () => {
    const { graph, outputNodeId } = buildSamGraph({
      applyPolygonRefinement: false,
      imageName: 'source.png',
      input: { prompt: '  cat  ', type: 'prompt' },
      invert: false,
      model: 'segment-anything-2-large',
    });

    expect(outputNodeId).toBe('sam-output');
    expect(graph).toEqual({
      edges: [
        edge('sam-source', 'image', 'sam-segment', 'image'),
        edge('sam-source', 'image', 'sam-detect', 'image'),
        edge('sam-detect', 'collection', 'sam-segment', 'bounding_boxes'),
        edge('sam-segment', 'mask', 'sam-output', 'mask'),
      ],
      id: 'select-object-sam',
      nodes: {
        ...baseNodes,
        'sam-detect': {
          detection_threshold: 0.3,
          id: 'sam-detect',
          is_intermediate: true,
          model: 'grounding-dino-base',
          prompt: 'cat',
          type: 'grounding_dino',
          use_cache: true,
        },
        'sam-segment': {
          apply_polygon_refinement: false,
          id: 'sam-segment',
          is_intermediate: true,
          mask_filter: 'largest',
          model: 'segment-anything-2-large',
          type: 'segment_anything',
          use_cache: true,
        },
      },
    });
  });

  it('builds the exact point-only visual graph with include and exclude labels', () => {
    const { graph } = buildSamGraph({
      applyPolygonRefinement: false,
      imageName: 'source.png',
      input: {
        bbox: null,
        excludePoints: [{ x: 7, y: 8 }],
        includePoints: [
          { x: 1, y: 2 },
          { x: 3, y: 4 },
        ],
        type: 'visual',
      },
      invert: false,
      model: 'segment-anything-2-large',
    });

    expect(graph.edges).toEqual([
      edge('sam-source', 'image', 'sam-segment', 'image'),
      edge('sam-segment', 'mask', 'sam-output', 'mask'),
    ]);
    expect(graph.nodes['sam-segment']).toEqual({
      apply_polygon_refinement: false,
      id: 'sam-segment',
      is_intermediate: true,
      mask_filter: 'largest',
      model: 'segment-anything-2-large',
      point_lists: [
        {
          points: [
            { label: 1, x: 1, y: 2 },
            { label: 1, x: 3, y: 4 },
            { label: -1, x: 7, y: 8 },
          ],
        },
      ],
      type: 'segment_anything',
      use_cache: true,
    });
  });

  it('builds the exact bbox-only visual graph', () => {
    const { graph } = buildSamGraph({
      applyPolygonRefinement: false,
      imageName: 'source.png',
      input: {
        bbox: { height: 20, width: 10, x: 2, y: 3 },
        excludePoints: [],
        includePoints: [],
        type: 'visual',
      },
      invert: false,
      model: 'segment-anything-2-large',
    });

    expect(graph.nodes['sam-segment']).toMatchObject({
      bounding_boxes: [{ x_max: 12, x_min: 2, y_max: 23, y_min: 3 }],
    });
    expect(graph.nodes['sam-segment']).not.toHaveProperty('point_lists');
  });

  it('aligns point and bbox lists for a combined visual graph', () => {
    const { graph } = buildSamGraph({
      applyPolygonRefinement: true,
      imageName: 'source.png',
      input: {
        bbox: { height: 4, width: 3, x: 2, y: 1 },
        excludePoints: [],
        includePoints: [{ x: 3, y: 2 }],
        type: 'visual',
      },
      invert: true,
      model: 'segment-anything-huge',
    });

    expect(graph.nodes['sam-segment']).toMatchObject({
      apply_polygon_refinement: true,
      bounding_boxes: [{ x_max: 5, x_min: 2, y_max: 5, y_min: 1 }],
      model: 'segment-anything-huge',
      point_lists: [{ points: [{ label: 1, x: 3, y: 2 }] }],
    });
    expect(graph.nodes['sam-output']).toMatchObject({ invert: true });
  });

  it.each([
    { prompt: '   ', type: 'prompt' } as const,
    { bbox: null, excludePoints: [], includePoints: [], type: 'visual' } as const,
    {
      bbox: { height: 2, width: -1, x: 0, y: 0 },
      excludePoints: [],
      includePoints: [],
      type: 'visual',
    } as const,
  ])('rejects invalid input $type', (input) => {
    expect(() =>
      buildSamGraph({
        applyPolygonRefinement: false,
        imageName: 'source.png',
        input,
        invert: false,
        model: 'segment-anything-2-large',
      })
    ).toThrow(input.type === 'prompt' ? 'A Segment Anything object prompt is required.' : 'visual input is required');
  });

  it.each([
    {
      bbox: null,
      excludePoints: [],
      includePoints: [{ x: 1.5, y: 2 }],
      type: 'visual',
    },
    {
      bbox: null,
      excludePoints: [{ x: Number.NaN, y: 2 }],
      includePoints: [],
      type: 'visual',
    },
    {
      bbox: null,
      excludePoints: [],
      includePoints: [{ x: 1, y: Number.POSITIVE_INFINITY }],
      type: 'visual',
    },
    {
      bbox: { height: 2, width: 1.5, x: 0, y: 0 },
      excludePoints: [],
      includePoints: [],
      type: 'visual',
    },
    {
      bbox: { height: 2, width: 1, x: Number.NEGATIVE_INFINITY, y: 0 },
      excludePoints: [],
      includePoints: [],
      type: 'visual',
    },
    {
      bbox: { height: 2, width: 1.5, x: 0, y: 0 },
      excludePoints: [],
      includePoints: [{ x: 1, y: 2 }],
      type: 'visual',
    },
    {
      bbox: { height: 2, width: Number.MAX_VALUE, x: Number.MAX_VALUE, y: 0 },
      excludePoints: [],
      includePoints: [],
      type: 'visual',
    },
  ] as const)('rejects backend-invalid coordinates %#', (input) => {
    expect(isSamInputValid(input)).toBe(false);
    expect(() =>
      buildSamGraph({
        applyPolygonRefinement: false,
        imageName: 'source.png',
        input,
        invert: false,
        model: 'segment-anything-2-large',
      })
    ).toThrow('visual input is required');
  });

  it('rejects a non-finite document bbox even when a finite point is present', () => {
    expect(
      isSamDocumentInputValid({
        bbox: { height: 2, width: Number.POSITIVE_INFINITY, x: 0, y: 0 },
        excludePoints: [],
        includePoints: [{ x: 1.25, y: 2.75 }],
        type: 'visual',
      })
    ).toBe(false);
  });
});

describe('documentToExportLocalSamInput', () => {
  it('subtracts a nonzero export origin, rounds finite points, and discards points outside the export', () => {
    expect(
      documentToExportLocalSamInput(
        {
          bbox: null,
          excludePoints: [{ x: 111.6, y: 210.4 }],
          includePoints: [
            { x: 99, y: 199 },
            { x: 105.4, y: 205.6 },
          ],
          type: 'visual',
        },
        { height: 10, width: 12, x: 100, y: 200 }
      )
    ).toEqual({
      bbox: null,
      excludePoints: [],
      includePoints: [{ x: 5, y: 6 }],
      type: 'visual',
    });
  });

  it('keeps inside edge points while discarding non-finite and outside points', () => {
    expect(
      documentToExportLocalSamInput(
        {
          bbox: null,
          excludePoints: [
            { x: Number.NaN, y: 10 },
            { x: 20, y: Number.POSITIVE_INFINITY },
          ],
          includePoints: [
            { x: 10, y: 20 },
            { x: 21.9, y: 29.9 },
            { x: 22, y: 25 },
          ],
          type: 'visual',
        },
        { height: 10, width: 12, x: 10, y: 20 }
      )
    ).toEqual({
      bbox: null,
      excludePoints: [],
      includePoints: [
        { x: 0, y: 0 },
        { x: 11, y: 9 },
      ],
      type: 'visual',
    });
  });

  it('clips bbox edges in export-local coordinates', () => {
    expect(
      documentToExportLocalSamInput(
        {
          bbox: { height: 30, width: 40, x: 90, y: 190 },
          excludePoints: [],
          includePoints: [],
          type: 'visual',
        },
        { height: 20, width: 30, x: 100, y: 200 }
      )
    ).toEqual({
      bbox: { height: 20, width: 30, x: 0, y: 0 },
      excludePoints: [],
      includePoints: [],
      type: 'visual',
    });
  });

  it('drops a bbox fully outside the export and leaves prompt input unchanged', () => {
    const prompt = { prompt: 'cat', type: 'prompt' } as const;
    expect(documentToExportLocalSamInput(prompt, { height: 20, width: 30, x: 100, y: 200 })).toBe(prompt);
    const converted = documentToExportLocalSamInput(
      {
        bbox: { height: 5, width: 5, x: 10, y: 20 },
        excludePoints: [],
        includePoints: [],
        type: 'visual',
      },
      { height: 20, width: 30, x: 100, y: 200 }
    );
    expect(converted.type).toBe('visual');
    if (converted.type !== 'visual') {
      throw new Error('Expected visual SAM input.');
    }
    expect(converted.bbox).toBeNull();
    expect(isSamInputValid(converted)).toBe(false);
  });

  it.each([
    { height: 0, width: 30, x: 100, y: 200 },
    { height: 20, width: 0, x: 100, y: 200 },
  ])('produces an invalid empty visual input for a zero-size export %#', (exportRect) => {
    const converted = documentToExportLocalSamInput(
      {
        bbox: null,
        excludePoints: [{ x: 101, y: 201 }],
        includePoints: [],
        type: 'visual',
      },
      exportRect
    );

    expect(converted).toEqual({ bbox: null, excludePoints: [], includePoints: [], type: 'visual' });
    expect(isSamInputValid(converted)).toBe(false);
  });
});
