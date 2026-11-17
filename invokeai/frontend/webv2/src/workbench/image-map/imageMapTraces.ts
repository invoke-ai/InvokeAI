import type { GalleryItemKey } from '@features/gallery/contracts';
import type { Layout } from 'plotly.js';

import type { ImageMapPoint } from './api';
import type { AxisRanges } from './imageMapViewport';

import { CLUSTER_PALETTE, getClusterColor } from './clusterPalette';

/** Trace builders in PhotoMapAI's fixed order; Current Image last, so gold markers draw on top. */

export const ALL_POINTS_TRACE = 'All Points';
export const CLUSTER_SELECTION_IMAGES_TRACE = 'Cluster Selection';
export const CLUSTER_SELECTION_VIDEOS_TRACE = 'Cluster Selection Videos';
export const HIGHLIGHTED_POINTS_TRACE = 'Highlighted Points';
export const CURRENT_IMAGE_TRACE = 'Current Image';

/** DBSCAN noise is cluster -1. */
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
    symbol?: string | string[];
    line?: { color: string; width: number };
  };
}

const IMAGE_SYMBOL = 'circle';
const VIDEO_SYMBOL = 'diamond';

/**
 * One trace per distinct appearance: per-point marker arrays made each zoom relayout ~14x slower at 170k points.
 * Noise is emitted first so dimmed points sit underneath. Hover gets slower (~1.5ms to ~7ms at 170k, kd-trees only for
 * 100k+ point traces); below ~5k points the per-trace overhead is a small accepted loss.
 */
export const buildAllPointsTraces = (points: ImageMapPoint[]): ScatterTrace[] => {
  // Bucketed by a small integer, not a composed string key, so the per-point pass does not allocate.
  const buckets: ImageMapPoint[][] = [];

  for (const point of points) {
    const cluster = clusterOf(point);
    const colorSlot = cluster < 0 ? CLUSTER_PALETTE.length : cluster % CLUSTER_PALETTE.length;
    const slot = colorSlot * 2 + (point.item.kind === 'video' ? 1 : 0);

    (buckets[slot] ??= []).push(point);
  }

  const traces: ScatterTrace[] = [];
  const order = [...buckets.keys()].sort(
    (left, right) => Number(right >= CLUSTER_PALETTE.length * 2) - Number(left >= CLUSTER_PALETTE.length * 2)
  );

  for (const slot of order) {
    const group = buckets[slot];

    if (group === undefined || group.length === 0) {
      continue;
    }

    const first = group[0]!;
    const cluster = clusterOf(first);

    traces.push({
      // Plotly carries strings, and a bare name would not say whether it names an image or a video.
      customdata: group.map((point) => point.key),
      hoverinfo: 'none',
      marker: {
        color: getClusterColor(cluster),
        opacity: cluster < 0 ? NOISE_OPACITY : POINT_OPACITY,
        size: 5,
        // Color already carries the cluster, so kind takes the shape channel.
        symbol: first.item.kind === 'video' ? VIDEO_SYMBOL : IMAGE_SYMBOL,
      },
      mode: 'markers',
      name: ALL_POINTS_TRACE,
      type: 'scattergl',
      x: group.map((point) => point.x),
      y: group.map((point) => point.y),
    });
  }

  return traces;
};

/** The bucket slot is an array index; the client does not validate the `int` the endpoint declares. */
const clusterOf = (point: ImageMapPoint): number => (Number.isInteger(point.cluster) ? point.cluster : -1);

/** Kept so a dimmed map can be restored exactly. */
export interface PointAppearance {
  color: string;
  opacity: number;
}

export const getTraceAppearance = (trace: ScatterTrace): PointAppearance => ({
  color: trace.marker.color as string,
  opacity: trace.marker.opacity as number,
});

// Opacity alone does not dim: dense clusters stack translucent markers back up to near full colour.
const DIM_TARGET = [110, 110, 116] as const;
const DIM_DESATURATION = 0.85;
const DIM_OPACITY = 0.18;

const mixTowardDimTarget = (hex: string): string => {
  const channels = [1, 3, 5].map((start, index) => {
    const value = parseInt(hex.slice(start, start + 2), 16);

    return Math.round(value + (DIM_TARGET[index]! - value) * DIM_DESATURATION);
  });

  return `#${channels.map((channel) => channel.toString(16).padStart(2, '0')).join('')}`;
};

export const dimAppearance = ({ color, opacity }: PointAppearance): PointAppearance => ({
  color: mixTowardDimTarget(color),
  opacity: DIM_OPACITY * (opacity / POINT_OPACITY),
});

/** One scalar per trace: per-point arrays would undo the zoom win. */
export const toBaseAppearanceRestyle = (
  appearances: readonly PointAppearance[],
  dimmed: boolean
): Record<string, unknown[]> => {
  const applied = dimmed ? appearances.map(dimAppearance) : appearances;

  return {
    'marker.color': applied.map((appearance) => appearance.color),
    'marker.opacity': applied.map((appearance) => appearance.opacity),
  };
};

/** A refresh can renumber clusters, so the colour is read from the points now drawn. */
export const getDominantCluster = (points: ImageMapPoint[], keys: ReadonlySet<GalleryItemKey>): number | null => {
  const counts = new Map<number, number>();

  for (const point of points) {
    if (keys.has(point.key)) {
      const cluster = clusterOf(point);

      counts.set(cluster, (counts.get(cluster) ?? 0) + 1);
    }
  }

  let dominant: number | null = null;
  let best = 0;

  for (const [cluster, count] of counts) {
    if (count > best) {
      best = count;
      dominant = cluster;
    }
  }

  return dominant;
};

/** Every marker property scalar, so a 100k-member cluster zooms like the base points; hits fall through to them. */
export const buildClusterSelectionTraces = (
  points: ImageMapPoint[],
  keys: ReadonlySet<GalleryItemKey>
): [ScatterTrace, ScatterTrace] => {
  const dominant = keys.size > 0 ? getDominantCluster(points, keys) : null;
  const color = getClusterColor(dominant ?? -1);
  const images: ImageMapPoint[] = [];
  const videos: ImageMapPoint[] = [];

  if (dominant !== null) {
    for (const point of points) {
      if (keys.has(point.key)) {
        (point.item.kind === 'video' ? videos : images).push(point);
      }
    }
  }

  const build = (members: ImageMapPoint[], name: string, symbol: string): ScatterTrace => ({
    customdata: members.map((point) => point.key),
    hoverinfo: 'skip',
    marker: { color, opacity: 1, size: 5, symbol },
    mode: 'markers',
    name,
    type: 'scattergl',
    x: members.map((point) => point.x),
    y: members.map((point) => point.y),
  });

  return [
    build(images, CLUSTER_SELECTION_IMAGES_TRACE, IMAGE_SYMBOL),
    build(videos, CLUSTER_SELECTION_VIDEOS_TRACE, VIDEO_SYMBOL),
  ];
};

export const toClusterSelectionRestyle = (traces: readonly ScatterTrace[]): Record<string, unknown[]> => ({
  customdata: traces.map((trace) => trace.customdata),
  'marker.color': traces.map((trace) => trace.marker.color),
  x: traces.map((trace) => trace.x),
  y: traces.map((trace) => trace.y),
});

/** Hand-made multi-selections only: per-point arrays are affordable only at hand-picked sizes. */
export const buildHighlightedPointsTrace = (
  points: ImageMapPoint[],
  selectedKeys: ReadonlySet<GalleryItemKey>
): ScatterTrace => {
  const selected = selectedKeys.size >= 2 ? points.filter((point) => selectedKeys.has(point.key)) : [];

  return {
    customdata: selected.map((point) => point.key),
    // Hit-testing falls through to the base points with the same customdata.
    hoverinfo: 'skip',
    marker: {
      color: selected.map((point) => getClusterColor(point.cluster)),
      line: { color: '#FFFFFF', width: 1 },
      opacity: 1,
      symbol: selected.map((point) => (point.item.kind === 'video' ? VIDEO_SYMBOL : IMAGE_SYMBOL)),
      size: 8,
    },
    mode: 'markers',
    name: HIGHLIGHTED_POINTS_TRACE,
    type: 'scattergl',
    x: selected.map((point) => point.x),
    y: selected.map((point) => point.y),
  };
};

export const buildCurrentImageTrace = (): ScatterTrace => ({
  customdata: [],
  // 'skip' (not 'none') drops the marker from hit-testing, so clicks and hovers land on the data point it covers.
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

const GRID_LINE_COLOR = 'rgba(128, 128, 128, 0.16)';
const GRID_ZERO_COLOR = 'rgba(128, 128, 128, 0.32)';

/** Restyle every per-point array: Plotly keeps omitted ones, mismatching scattergl lengths. */
export const toHighlightRestyle = (trace: ScatterTrace): Record<string, unknown[]> => ({
  customdata: [trace.customdata],
  'marker.color': [trace.marker.color as string[]],
  'marker.symbol': [trace.marker.symbol as string[]],
  x: [trace.x],
  y: [trace.y],
});

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
  uirevision: 'image-map',
  xaxis: {
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

/** Largest, then id, wins decluttering. */
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

  return [...sums.entries()]
    .sort(([clusterA, a], [clusterB, b]) => b.count - a.count || clusterA - clusterB)
    .map(([cluster, { count, x, maxY }]) => ({
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

// Labels cannot be measured before drawing, so width is estimated from fixed-10px text length.
const LABEL_CHAR_WIDTH_PX = 6;
const LABEL_HEIGHT_PX = 18;
const LABEL_GAP_PX = 4;
/** 18px marker, 2px outline and one pixel of label clearance. */
const MARKER_RADIUS_PX = 11;

interface LabelRect {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

const rectsCollide = (a: LabelRect, b: LabelRect): boolean =>
  a.left < b.right + LABEL_GAP_PX &&
  a.right > b.left - LABEL_GAP_PX &&
  a.top < b.bottom + LABEL_GAP_PX &&
  a.bottom > b.top - LABEL_GAP_PX;

/** Onscreen only: offscreen reservations churn hidden labels while panning. Bounds checks reject NaN. */
const markerRectFor = (
  markerPoint: { x: number; y: number },
  ranges: AxisRanges,
  spanX: number,
  spanY: number,
  widthPx: number,
  heightPx: number
): LabelRect | null => {
  const centerX = ((markerPoint.x - ranges.x[0]) / spanX) * widthPx;
  const centerY = ((ranges.y[1] - markerPoint.y) / spanY) * heightPx;
  const onScreen =
    centerX >= -MARKER_RADIUS_PX &&
    centerX <= widthPx + MARKER_RADIUS_PX &&
    centerY >= -MARKER_RADIUS_PX &&
    centerY <= heightPx + MARKER_RADIUS_PX;

  return onScreen
    ? {
        bottom: centerY + MARKER_RADIUS_PX,
        left: centerX - MARKER_RADIUS_PX,
        right: centerX + MARKER_RADIUS_PX,
        top: centerY - MARKER_RADIUS_PX,
      }
    : null;
};

/** Current-image markers outrank labels: Plotly SVG annotations render above WebGL and cannot go underneath. */
export const declutterAnnotations = (
  annotations: ClusterAnnotation[],
  ranges: AxisRanges,
  widthPx: number,
  heightPx: number,
  markerPoint?: { x: number; y: number } | null
): ClusterAnnotation[] => {
  const spanX = ranges.x[1] - ranges.x[0];
  const spanY = ranges.y[1] - ranges.y[0];

  if (spanX <= 0 || spanY <= 0 || widthPx <= 0 || heightPx <= 0) {
    return annotations;
  }

  const kept: ClusterAnnotation[] = [];
  const blockingRects: LabelRect[] = [];
  const markerRect = markerPoint ? markerRectFor(markerPoint, ranges, spanX, spanY, widthPx, heightPx) : null;

  for (const annotation of annotations) {
    const centerX = ((annotation.x - ranges.x[0]) / spanX) * widthPx;
    // Screen y grows downward: yanchor 'bottom' + positive yshift puts the pill's bottom yshift px above the anchor.
    const bottom = ((ranges.y[1] - annotation.y) / spanY) * heightPx - annotation.yshift;
    const halfWidth = (annotation.text.length * LABEL_CHAR_WIDTH_PX) / 2 + annotation.borderpad;
    const rect: LabelRect = {
      bottom,
      left: centerX - halfWidth,
      right: centerX + halfWidth,
      top: bottom - LABEL_HEIGHT_PX,
    };

    if (blockingRects.some((other) => rectsCollide(rect, other))) {
      continue;
    }

    // Reserve before testing marker overlap, so lower-priority labels cannot take the cleared spot.
    blockingRects.push(rect);

    if (markerRect !== null && rectsCollide(rect, markerRect)) {
      continue;
    }

    kept.push(annotation);
  }

  return kept;
};
