import type { Rect, Vec2 } from '@workbench/canvas-engine/types';
import type { BackendGraphContract } from '@workbench/types';

import { addEdge, addNode } from '@workbench/generation/graphBuilder';

export type SamModel = 'segment-anything-2-large' | 'segment-anything-huge';

export type SamInput =
  | { type: 'prompt'; prompt: string }
  | {
      type: 'visual';
      includePoints: readonly Vec2[];
      excludePoints: readonly Vec2[];
      bbox?: Rect | null;
    };

export interface BuildSamGraphOptions {
  imageName: string;
  input: SamInput;
  model: SamModel;
  invert: boolean;
  applyPolygonRefinement: boolean;
}

export interface BuiltSamGraph {
  graph: BackendGraphContract;
  outputNodeId: string;
}

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const isFinitePoint = ({ x, y }: Vec2): boolean => Number.isFinite(x) && Number.isFinite(y);
const isIntegerPoint = (point: Vec2): boolean =>
  isFinitePoint(point) && Number.isInteger(point.x) && Number.isInteger(point.y);
const isFiniteBox = (bbox: Rect): boolean =>
  Number.isFinite(bbox.x) &&
  Number.isFinite(bbox.y) &&
  Number.isFinite(bbox.width) &&
  Number.isFinite(bbox.height) &&
  bbox.width > 0 &&
  bbox.height > 0 &&
  Number.isFinite(bbox.x + bbox.width) &&
  Number.isFinite(bbox.y + bbox.height) &&
  bbox.x + bbox.width > bbox.x &&
  bbox.y + bbox.height > bbox.y;
const isIntegerBox = (bbox: Rect): boolean =>
  isFiniteBox(bbox) &&
  Number.isInteger(bbox.x) &&
  Number.isInteger(bbox.y) &&
  Number.isInteger(bbox.width) &&
  Number.isInteger(bbox.height);

/** Validates document-space inputs before scheduling or stable hashing. */
export const isSamDocumentInputValid = (input: SamInput): boolean => {
  if (input.type === 'prompt') {
    return input.prompt.trim().length > 0;
  }
  const points = [...input.includePoints, ...input.excludePoints];
  const hasBox = input.bbox !== null && input.bbox !== undefined;
  const boxIsValid = input.bbox === null || input.bbox === undefined || isFiniteBox(input.bbox);
  return points.every(isFinitePoint) && boxIsValid && (points.length > 0 || hasBox);
};

export const isSamInputValid = (input: SamInput): boolean => {
  if (input.type === 'prompt') {
    return input.prompt.trim().length > 0;
  }
  const points = [...input.includePoints, ...input.excludePoints];
  const hasBox = input.bbox !== null && input.bbox !== undefined;
  const boxIsValid = input.bbox === null || input.bbox === undefined || isIntegerBox(input.bbox);
  return points.every(isIntegerPoint) && boxIsValid && (points.length > 0 || hasBox);
};

/** Converts document-space visual prompts to the pixel coordinates of a bounded layer export. */
export const documentToExportLocalSamInput = (input: SamInput, exportRect: Rect): SamInput => {
  if (input.type === 'prompt') {
    return input;
  }

  if (
    !Number.isFinite(exportRect.x) ||
    !Number.isFinite(exportRect.y) ||
    !Number.isInteger(exportRect.width) ||
    !Number.isInteger(exportRect.height) ||
    exportRect.width <= 0 ||
    exportRect.height <= 0
  ) {
    return { bbox: null, excludePoints: [], includePoints: [], type: 'visual' };
  }

  const convertPoint = ({ x, y }: Vec2): Vec2 => ({
    x: clamp(Math.round(x - exportRect.x), 0, exportRect.width - 1),
    y: clamp(Math.round(y - exportRect.y), 0, exportRect.height - 1),
  });
  const isPointInsideExport = ({ x, y }: Vec2): boolean =>
    isFinitePoint({ x, y }) &&
    x >= exportRect.x &&
    x < exportRect.x + exportRect.width &&
    y >= exportRect.y &&
    y < exportRect.y + exportRect.height;
  let bbox: Rect | null = null;
  if (input.bbox && isFiniteBox(input.bbox)) {
    const xMin = clamp(Math.round(input.bbox.x - exportRect.x), 0, exportRect.width);
    const yMin = clamp(Math.round(input.bbox.y - exportRect.y), 0, exportRect.height);
    const xMax = clamp(Math.round(input.bbox.x + input.bbox.width - exportRect.x), 0, exportRect.width);
    const yMax = clamp(Math.round(input.bbox.y + input.bbox.height - exportRect.y), 0, exportRect.height);
    if (xMax > xMin && yMax > yMin) {
      bbox = { height: yMax - yMin, width: xMax - xMin, x: xMin, y: yMin };
    }
  }

  return {
    bbox,
    excludePoints: input.excludePoints.filter(isPointInsideExport).map(convertPoint),
    includePoints: input.includePoints.filter(isPointInsideExport).map(convertPoint),
    type: 'visual',
  };
};

export const buildSamGraph = (options: BuildSamGraphOptions): BuiltSamGraph => {
  if (!isSamInputValid(options.input)) {
    throw new Error(
      options.input.type === 'prompt'
        ? 'A Segment Anything object prompt is required.'
        : 'A Segment Anything visual input is required.'
    );
  }

  const graph: BackendGraphContract = { edges: [], id: 'select-object-sam', nodes: {} };
  const source = addNode(graph, {
    id: 'sam-source',
    image: { image_name: options.imageName },
    type: 'image',
  });
  const points =
    options.input.type === 'visual'
      ? [
          ...options.input.includePoints.map(({ x, y }) => ({ label: 1, x, y })),
          ...options.input.excludePoints.map(({ x, y }) => ({ label: -1, x, y })),
        ]
      : [];
  const bbox = options.input.type === 'visual' ? options.input.bbox : null;
  const segment = addNode(graph, {
    ...(points.length > 0 ? { point_lists: [{ points }] } : {}),
    ...(bbox
      ? {
          bounding_boxes: [
            {
              x_max: bbox.x + bbox.width,
              x_min: bbox.x,
              y_max: bbox.y + bbox.height,
              y_min: bbox.y,
            },
          ],
        }
      : {}),
    apply_polygon_refinement: options.applyPolygonRefinement,
    id: 'sam-segment',
    mask_filter: 'largest',
    model: options.model,
    type: 'segment_anything',
  });
  const output = addNode(graph, {
    id: 'sam-output',
    image: { image_name: options.imageName },
    invert: options.invert,
    type: 'apply_tensor_mask_to_image',
  });

  addEdge(graph, source, 'image', segment, 'image');
  if (options.input.type === 'prompt') {
    const detect = addNode(graph, {
      detection_threshold: 0.3,
      id: 'sam-detect',
      model: 'grounding-dino-base',
      prompt: options.input.prompt.trim(),
      type: 'grounding_dino',
    });
    addEdge(graph, source, 'image', detect, 'image');
    addEdge(graph, detect, 'collection', segment, 'bounding_boxes');
  }
  addEdge(graph, segment, 'mask', output, 'mask');

  return { graph, outputNodeId: output.id };
};
