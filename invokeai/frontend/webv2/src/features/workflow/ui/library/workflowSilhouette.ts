import type { ProjectGraphState, WorkflowNode } from '@features/workflow/core/types';

/**
 * A workflow's shape for a thumbnail well that has no image: one rect per node, in fractions of a 3:2 well. Node
 * documents carry positions but no measured sizes, so each node kind gets a nominal size (invocation height grows
 * with its inputs). Large graphs keep their full extent but draw an evenly spaced, deterministic subset of nodes.
 */

export interface WorkflowSilhouetteRect {
  /** Left edge, as a fraction of the well's width. */
  x: number;
  /** Top edge, as a fraction of the well's height. */
  y: number;
  width: number;
  height: number;
}

/** Enough to read as a graph at card size; more only adds paint cost. */
export const MAX_SILHOUETTE_RECTS = 24;

/** The well the rects are laid out in, matching WORKFLOW_THUMBNAIL_ASPECT_RATIO. */
const WELL_ASPECT_RATIO = 3 / 2;
/** Space kept clear on every side, as a fraction of each well dimension. */
const PADDING = 0.12;
/** A single node must not fill the well. */
const MAX_SCALE_NODE_WIDTHS = 2.5;
/** Rects smaller than this vanish at card size. */
const MIN_RECT_FRACTION = 0.02;

const NOMINAL_NODE_WIDTH = 320;

const getNominalSize = (node: WorkflowNode): { width: number; height: number } => {
  switch (node.type) {
    case 'invocation':
      return {
        height: 72 + 28 * Math.min(Object.keys(node.data.inputs).length, 12),
        width: NOMINAL_NODE_WIDTH,
      };
    case 'notes':
      return { height: 160, width: NOMINAL_NODE_WIDTH };
    case 'current_image':
      return { height: NOMINAL_NODE_WIDTH, width: NOMINAL_NODE_WIDTH };
    case 'connector':
      return { height: 40, width: 40 };
  }
};

const isFinitePosition = (node: WorkflowNode): boolean =>
  Number.isFinite(node.position.x) && Number.isFinite(node.position.y);

/** Every `step`-th node in reading order, so the subset spreads across the graph and repeats for the same input. */
const sampleNodes = (nodes: readonly WorkflowNode[]): readonly WorkflowNode[] => {
  if (nodes.length <= MAX_SILHOUETTE_RECTS) {
    return nodes;
  }

  const ordered = [...nodes].sort(
    (left, right) =>
      left.position.x - right.position.x ||
      left.position.y - right.position.y ||
      (left.id < right.id ? -1 : left.id > right.id ? 1 : 0)
  );
  const step = ordered.length / MAX_SILHOUETTE_RECTS;

  return Array.from({ length: MAX_SILHOUETTE_RECTS }, (_, index) => ordered[Math.floor(index * step)]!);
};

/** Empty when the document has no positioned nodes; callers then show their own placeholder. */
export const layoutWorkflowSilhouette = (nodes: readonly WorkflowNode[]): WorkflowSilhouetteRect[] => {
  const positioned = nodes.filter(isFinitePosition);

  if (positioned.length === 0) {
    return [];
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const node of positioned) {
    const { height, width } = getNominalSize(node);

    minX = Math.min(minX, node.position.x);
    minY = Math.min(minY, node.position.y);
    maxX = Math.max(maxX, node.position.x + width);
    maxY = Math.max(maxY, node.position.y + height);
  }

  // Work in units where the well is WELL_ASPECT_RATIO wide and 1 high, then divide back into fractions.
  const availableWidth = WELL_ASPECT_RATIO * (1 - 2 * PADDING);
  const availableHeight = 1 - 2 * PADDING;
  const graphWidth = maxX - minX;
  const graphHeight = maxY - minY;
  const scale = Math.min(
    availableWidth / graphWidth,
    availableHeight / graphHeight,
    availableWidth / (MAX_SCALE_NODE_WIDTHS * NOMINAL_NODE_WIDTH)
  );
  const offsetX = (WELL_ASPECT_RATIO - graphWidth * scale) / 2;
  const offsetY = (1 - graphHeight * scale) / 2;

  return sampleNodes(positioned).map((node) => {
    const { height, width } = getNominalSize(node);

    return {
      height: Math.max(MIN_RECT_FRACTION, height * scale),
      width: Math.max(MIN_RECT_FRACTION, (width * scale) / WELL_ASPECT_RATIO),
      x: (offsetX + (node.position.x - minX) * scale) / WELL_ASPECT_RATIO,
      y: offsetY + (node.position.y - minY) * scale,
    };
  });
};

/** The image's coordinate space: the 3:2 well, 150 by 100 units. */
const IMAGE_WIDTH = 150;
const IMAGE_HEIGHT = 100;
/** A mid grey at low opacity reads on both the light and the dark well; a data URI cannot use theme variables. */
const RECT_STYLE = 'fill="rgb(140 144 156 / 0.16)" stroke="rgb(140 144 156 / 0.42)" stroke-width="0.6"';

export interface WorkflowSilhouetteImage {
  /** A CSS `url()` value holding the silhouette as an SVG data URI. */
  url: string;
  rectCount: number;
}

const round = (value: number) => Math.round(value * 10) / 10;

/** Absent when the document has nothing to draw. */
export const toWorkflowSilhouetteImage = (rects: readonly WorkflowSilhouetteRect[]): WorkflowSilhouetteImage | null => {
  if (rects.length === 0) {
    return null;
  }

  const shapes = rects
    .map(
      (rect) =>
        `<rect x="${round(rect.x * IMAGE_WIDTH)}" y="${round(rect.y * IMAGE_HEIGHT)}" width="${round(rect.width * IMAGE_WIDTH)}" height="${round(rect.height * IMAGE_HEIGHT)}" rx="1.5"/>`
    )
    .join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${IMAGE_WIDTH} ${IMAGE_HEIGHT}" preserveAspectRatio="none"><g ${RECT_STYLE}>${shapes}</g></svg>`;

  return { rectCount: rects.length, url: `url("data:image/svg+xml,${encodeURIComponent(svg)}")` };
};

/**
 * One image per parsed document. A document object stands for one revision of one workflow (an edit or a newer
 * revision parses a new one), and the cache lets go of it with the document, so a card scrolling back into view or
 * re-rendering never lays the graph out again.
 */
const images = new WeakMap<ProjectGraphState, WorkflowSilhouetteImage | null>();

export const getWorkflowSilhouetteImage = (document: ProjectGraphState): WorkflowSilhouetteImage | null => {
  let image = images.get(document);

  if (image === undefined) {
    image = toWorkflowSilhouetteImage(layoutWorkflowSilhouette(document.nodes));
    images.set(document, image);
  }

  return image;
};
