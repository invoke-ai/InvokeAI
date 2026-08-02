import type { Layout } from 'plotly.js';

import type { ImageMapPoint } from './api';
import type { AxisRanges } from './imageMapViewport';

import { getClusterColor } from './clusterPalette';

/**
 * Pure trace/layout builders for the map, kept apart from the plotly host so
 * the math is testable without WebGL. Trace identity follows PhotoMapAI:
 * named traces in a fixed z-order, with "Current Image" always last so the
 * gold marker renders on top.
 */

export const ALL_POINTS_TRACE = 'All Points';
export const HIGHLIGHTED_POINTS_TRACE = 'Highlighted Points';
export const CURRENT_IMAGE_TRACE = 'Current Image';

/** Opacity for DBSCAN noise points (cluster -1); clustered points are solid. */
const NOISE_OPACITY = 0.25;
const POINT_OPACITY = 0.85;

export interface ScatterTrace {
  x: number[];
  y: number[];
  customdata: string[];
  mode: 'markers';
  type: 'scattergl';
  name: string;
  hoverinfo: 'none' | 'skip' | 'text';
  text?: string[];
  marker: {
    color: string | string[];
    opacity: number | number[];
    size: number;
    symbol?: string;
    line?: { color: string; width: number };
  };
}

export const buildAllPointsTrace = (points: ImageMapPoint[]): ScatterTrace => ({
  customdata: points.map((point) => point.imageName),
  hoverinfo: 'none',
  marker: {
    color: points.map((point) => getClusterColor(point.cluster)),
    opacity: points.map((point) => (point.cluster < 0 ? NOISE_OPACITY : POINT_OPACITY)),
    size: 5,
  },
  mode: 'markers',
  name: ALL_POINTS_TRACE,
  type: 'scattergl',
  x: points.map((point) => point.x),
  y: points.map((point) => point.y),
});

/**
 * The gallery's multi-selection (e.g. a cluster click), drawn larger with a
 * white outline over the base points. Empty when fewer than two images are
 * selected — a single selection is already marked by the gold target.
 */
export const buildHighlightedPointsTrace = (
  points: ImageMapPoint[],
  selectedNames: ReadonlySet<string>
): ScatterTrace => {
  const selected = selectedNames.size >= 2 ? points.filter((point) => selectedNames.has(point.imageName)) : [];

  return {
    customdata: selected.map((point) => point.imageName),
    // 'skip': highlighted points sit over their base points, which carry the
    // same customdata — hit-testing should fall through to them.
    hoverinfo: 'skip',
    marker: {
      color: selected.map((point) => getClusterColor(point.cluster)),
      line: { color: '#FFFFFF', width: 1 },
      opacity: 1,
      size: 8,
    },
    mode: 'markers',
    name: HIGHLIGHTED_POINTS_TRACE,
    type: 'scattergl',
    x: selected.map((point) => point.x),
    y: selected.map((point) => point.y),
  };
};

/**
 * The gold target marking the current gallery image. Built empty here; a later
 * PR restyles it live as the selection changes.
 */
export const buildCurrentImageTrace = (): ScatterTrace => ({
  customdata: [],
  // 'skip' (not 'none') excludes the marker from hit-testing entirely, so
  // clicks and hovers land on the underlying data point it covers.
  hoverinfo: 'skip',
  marker: {
    color: '#FFD700',
    line: { color: '#000000', width: 2 },
    opacity: 1,
    size: 18,
    symbol: 'circle-dot',
  },
  mode: 'markers',
  name: CURRENT_IMAGE_TRACE,
  type: 'scattergl',
  x: [],
  y: [],
});

// Theme-independent grid: visible but recessive on both light and dark
// map backgrounds; the axis origin is slightly stronger for orientation.
const GRID_LINE_COLOR = 'rgba(128, 128, 128, 0.16)';
const GRID_ZERO_COLOR = 'rgba(128, 128, 128, 0.32)';

export const buildMapLayout = (
  initialRanges?: AxisRanges | null,
  annotations: ClusterAnnotation[] = []
): Partial<Layout> => ({
  annotations: annotations as unknown as Layout['annotations'],
  dragmode: 'pan',
  margin: { b: 0, l: 0, r: 0, t: 0 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  showlegend: false,
  // Preserves the user's pan/zoom across Plotly.react data updates.
  uirevision: 'image-map',
  xaxis: {
    // Unlabeled gridlines give the eye a frame of reference when the map is
    // sparse. Mid-gray at low alpha reads on every theme background, like
    // the rest of the map's fixed styling.
    gridcolor: GRID_LINE_COLOR,
    gridwidth: 1,
    range: initialRanges?.x,
    scaleanchor: 'y',
    showgrid: true,
    showticklabels: false,
    zeroline: true,
    zerolinecolor: GRID_ZERO_COLOR,
    zerolinewidth: 1,
  },
  yaxis: {
    gridcolor: GRID_LINE_COLOR,
    gridwidth: 1,
    range: initialRanges?.y,
    showgrid: true,
    showticklabels: false,
    zeroline: true,
    zerolinecolor: GRID_ZERO_COLOR,
    zerolinewidth: 1,
  },
});

export interface ClusterAnnotation {
  x: number;
  y: number;
  text: string;
  showarrow: false;
  font: { color: string; size: number };
  bgcolor: string;
  borderpad: number;
  opacity: number;
  xanchor: 'center';
  yanchor: 'bottom';
  yshift: number;
}

/**
 * One text annotation per labeled cluster: centered horizontally on the
 * cluster but anchored just above its topmost point (a fixed pixel lift, so
 * zoom never lands the label on the points it names). White-on-dark pill
 * styling is theme-independent — it reads on every map background.
 * Pure so placement math is testable; plotly consumes the array via layout.
 */
export const buildClusterAnnotations = (
  points: ImageMapPoint[],
  labelsByCluster: Record<string, string> | null
): ClusterAnnotation[] => {
  if (!labelsByCluster) {
    return [];
  }

  const sums = new Map<number, { count: number; x: number; maxY: number }>();

  for (const point of points) {
    if (point.cluster < 0 || !(String(point.cluster) in labelsByCluster)) {
      continue;
    }

    const entry = sums.get(point.cluster) ?? { count: 0, maxY: -Infinity, x: 0 };
    entry.count += 1;
    entry.x += point.x;
    entry.maxY = Math.max(entry.maxY, point.y);
    sums.set(point.cluster, entry);
  }

  return [...sums.entries()].map(([cluster, { count, x, maxY }]) => ({
    bgcolor: 'rgba(0,0,0,0.65)',
    borderpad: 2,
    font: { color: '#FFFFFF', size: 10 },
    opacity: 1,
    showarrow: false,
    text: labelsByCluster[String(cluster)],
    x: x / count,
    xanchor: 'center',
    y: maxY,
    yanchor: 'bottom',
    yshift: 8,
  }));
};
