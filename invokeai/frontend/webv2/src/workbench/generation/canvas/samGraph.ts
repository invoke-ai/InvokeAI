import type { BackendGraphContract } from '@workbench/types';

import { addEdge, addNode } from '@workbench/generation/graphBuilder';

export type SamModel = 'segment-anything-2-large' | 'segment-anything-huge';

export interface BuildSamGraphOptions {
  imageName: string;
  prompt: string;
  model: SamModel;
  applyPolygonRefinement: boolean;
}

export interface BuiltSamGraph {
  graph: BackendGraphContract;
  outputNodeId: string;
}

export const buildSamGraph = (options: BuildSamGraphOptions): BuiltSamGraph => {
  const prompt = options.prompt.trim();
  if (!prompt) {
    throw new Error('A Segment Anything object prompt is required.');
  }

  const graph: BackendGraphContract = { edges: [], id: 'select-object-sam', nodes: {} };
  const source = addNode(graph, {
    id: 'sam-source',
    image: { image_name: options.imageName },
    type: 'image',
  });
  const detect = addNode(graph, {
    detection_threshold: 0.3,
    id: 'sam-detect',
    model: 'grounding-dino-base',
    prompt,
    type: 'grounding_dino',
  });
  const segment = addNode(graph, {
    apply_polygon_refinement: options.applyPolygonRefinement,
    id: 'sam-segment',
    mask_filter: 'largest',
    model: options.model,
    type: 'segment_anything',
  });
  const output = addNode(graph, {
    id: 'sam-output',
    image: { image_name: options.imageName },
    invert: false,
    type: 'apply_tensor_mask_to_image',
  });

  addEdge(graph, source, 'image', detect, 'image');
  addEdge(graph, source, 'image', segment, 'image');
  addEdge(graph, detect, 'collection', segment, 'bounding_boxes');
  addEdge(graph, segment, 'mask', output, 'mask');

  return { graph, outputNodeId: output.id };
};
