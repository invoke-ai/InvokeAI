import type { GalleryItemKey } from '@features/gallery/contracts';
import type { Layout } from 'plotly.js';

import type { ImageMapPoint } from './api';
import type { AxisRanges } from './imageMapViewport';

import { CLUSTER_PALETTE, getClusterColor } from './clusterPalette';

/**
 * Pure WebGL-independent trace builders follow PhotoMapAI's fixed named order, placing Current Image last so gold
 * markers render above other traces.
 */

export const ALL_POINTS_TRACE = 'All Points';
export const CLUSTER_SELECTION_IMAGES_TRACE = 'Cluster Selection';
export const CLUSTER_SELECTION_VIDEOS_TRACE = 'Cluster Selection Videos';
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
    symbol?: string | string[];
    line?: { color: string; width: number };
  };
}

/** Marker shapes per media kind; see `buildAllPointsTraces`. */
const IMAGE_SYMBOL = 'circle';
const VIDEO_SYMBOL = 'diamond';

/**
 * The base points, split into one trace per distinct appearance.
 *
 * A single trace carrying per-point `color`/`opacity`/`symbol` arrays is what
 * made zooming a large map slow: plotly reprocesses every one of those arrays
 * on each relayout, and a zoom is a relayout per frame. Splitting the points
 * so every marker property is scalar cuts a zoom step by roughly 14x at 170k
 * points and 20x at 300k, and builds the scene about twice as fast while
 * retaining half the heap.
 *
 * Each point keeps the appearance it had, but the ORDER changes: points now
 * paint grouped rather than in gallery order, so where two points overlap the
 * one on top can differ. Markers are translucent, so that is visible. Noise
 * is emitted first, which is the deliberate part — dimmed points belong under
 * the clustered ones. Within a group the original point order is preserved.
 *
 * The cost is hover. Plotly builds a kd-tree for hit-testing only on traces
 * of 100k points or more, so one 170k trace had one and none of these do:
 * hover goes from ~1.5ms to ~7ms at 170k and ~11ms at 300k, throttled to
 * 20/s. Zoom was 180ms+ at those sizes, so this trades a cost nobody could
 * work through for one that is merely warm.
 *
 * At most (palette + noise) x (image, video) groups exist, so the trace count
 * is bounded by the palette rather than by the gallery. Below roughly 5k
 * points the per-trace overhead makes this a small net loss (~2.5ms a frame
 * at 200 points) — accepted rather than switched on a threshold, because
 * making the draw order depend on gallery size is a worse bargain than the
 * milliseconds.
 */
export const buildAllPointsTraces = (points: ImageMapPoint[]): ScatterTrace[] => {
  // Bucketed by a small integer rather than a composed string key: this runs
  // over every point on each data change, and building a key per point is the
  // one part of it that would allocate.
  const buckets: ImageMapPoint[][] = [];

  for (const point of points) {
    const cluster = clusterOf(point);
    const colorSlot = cluster < 0 ? CLUSTER_PALETTE.length : cluster % CLUSTER_PALETTE.length;
    const slot = colorSlot * 2 + (point.item.kind === 'video' ? 1 : 0);

    (buckets[slot] ??= []).push(point);
  }

  const traces: ScatterTrace[] = [];
  // Noise occupies the slots just past the palette, and emitting it first is
  // what puts the dimmed points underneath.
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
      // The gallery's item key, so a click or hover resolves back to the kind
      // the point stands for — plotly carries strings, and a bare name would
      // not say whether it names an image or a video.
      customdata: group.map((point) => point.key),
      hoverinfo: 'none',
      marker: {
        color: getClusterColor(cluster),
        opacity: cluster < 0 ? NOISE_OPACITY : POINT_OPACITY,
        size: 5,
        // Videos are diamonds. Color already carries the cluster, so kind
        // needs the one remaining channel: without it a clip is
        // pixel-identical to an image and can only be found by hovering
        // points one at a time.
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

/**
 * The point's DBSCAN label, or noise for anything that cannot be one.
 *
 * The bucket slot below is an array index, so a non-integer label would write
 * a string property that `buckets.keys()` never yields — those points would
 * silently never reach a trace, and a fractional one would collide with
 * another cluster's slot and take its colour. The endpoint declares `int` and
 * the client does not validate the body, so this is the guard that keeps a
 * contract slip from quietly deleting part of the map.
 */
const clusterOf = (point: ImageMapPoint): number => (Number.isInteger(point.cluster) ? point.cluster : -1);

/** A base trace's own look, kept so a dimmed map can be restored exactly. */
export interface PointAppearance {
  color: string;
  opacity: number;
}

export const getTraceAppearance = (trace: ScatterTrace): PointAppearance => ({
  color: trace.marker.color as string,
  opacity: trace.marker.opacity as number,
});

// Dimmed points keep a trace of their hue — enough to read the map's structure,
// not enough to compete with the selected cluster. Opacity alone was not: dense
// clusters stack translucent markers back up to near full colour.
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

/** How a base trace looks while a cluster is selected; noise stays fainter than clustered points. */
export const dimAppearance = ({ color, opacity }: PointAppearance): PointAppearance => ({
  color: mixTowardDimTarget(color),
  opacity: DIM_OPACITY * (opacity / POINT_OPACITY),
});

/**
 * Restyle for the base traces, dimmed or restored. One scalar per trace: a per-point array here would put back the
 * zoom cost the per-appearance split removed.
 */
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

/**
 * The cluster most of these items belong to on the current map. A refresh can renumber clusters after the selection
 * was made, so the colour is read from the points now drawn rather than remembered from the click.
 */
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

/**
 * A selected cluster redrawn at full colour over the dimmed map: one trace per kind, every marker property scalar, so
 * a 100k-member cluster zooms like the base points do. Hit-testing falls through to the base points underneath.
 */
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

/** In-place update for both cluster-selection traces, in trace order. */
export const toClusterSelectionRestyle = (traces: readonly ScatterTrace[]): Record<string, unknown[]> => ({
  customdata: traces.map((trace) => trace.customdata),
  'marker.color': traces.map((trace) => trace.marker.color),
  x: traces.map((trace) => trace.x),
  y: traces.map((trace) => trace.y),
});

/**
 * White-outlined enlarged trace for hand-made multi-selections; fewer than two items need only the gold current
 * target. Cluster clicks use the cluster-selection traces, since per-point arrays are affordable only at hand-picked
 * sizes.
 */
export const buildHighlightedPointsTrace = (
  points: ImageMapPoint[],
  selectedKeys: ReadonlySet<GalleryItemKey>
): ScatterTrace => {
  const selected = selectedKeys.size >= 2 ? points.filter((point) => selectedKeys.has(point.key)) : [];

  return {
    customdata: selected.map((point) => point.key),
    // 'skip': highlighted points sit over their base points, which carry the
    // same customdata — hit-testing should fall through to them.
    hoverinfo: 'skip',
    marker: {
      color: selected.map((point) => getClusterColor(point.cluster)),
      line: { color: '#FFFFFF', width: 1 },
      opacity: 1,
      // The overlay sits on top of the base point, so it carries the kind's
      // shape too — otherwise selecting a video turns its diamond into a circle.
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

/** Gold current-item target, populated by live selection restyles. */
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

// Theme-independent grid with a stronger origin for orientation.
const GRID_LINE_COLOR = 'rgba(128, 128, 128, 0.16)';
const GRID_ZERO_COLOR = 'rgba(128, 128, 128, 0.32)';

/**
 * Restyle every per-point array derived from the trace. Plotly retains omitted arrays, causing scattergl length
 * mismatches after selection changes.
 */
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
  // Preserves the user's pan/zoom across Plotly.react data updates.
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

/**
 * Place theme-independent cluster pills above topmost points with fixed-pixel clearance. Sort largest clusters
 * first, then id, defining declutter priority.
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

// Estimate SVG label collision width from fixed-10px text length because labels cannot be measured before drawing.
const LABEL_CHAR_WIDTH_PX = 6;
const LABEL_HEIGHT_PX = 18;
/** Two labels closer than this (edge to edge) count as colliding. */
const LABEL_GAP_PX = 4;
/** Marker half-extent includes its 18px size, 2px outline and one extra pixel of label clearance. */
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

/**
 * Return onscreen marker footprint only; offscreen reservations churn invisible labels during panning. Bounds
 * checks reject NaN too.
 */
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

/**
 * Greedily retain noncolliding labels in priority order. Current-image markers outrank labels because Plotly SVG
 * annotations render above WebGL and cannot be stacked underneath the gold target.
 */
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
  // Footprints later labels must avoid: every label that won its spot,
  // including the ones the marker then hid (see below).
  const blockingRects: LabelRect[] = [];
  const markerRect = markerPoint ? markerRectFor(markerPoint, ranges, spanX, spanY, widthPx, heightPx) : null;

  for (const annotation of annotations) {
    const centerX = ((annotation.x - ranges.x[0]) / spanX) * widthPx;
    // Screen y grows downward; yanchor 'bottom' + positive yshift puts the
    // pill's bottom edge yshift pixels above the anchor point.
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

    // Reserve a winning label's space before testing marker overlap, preventing lower-priority replacements in the
    // cleared region. Labels already losing to another label reserve nothing.
    blockingRects.push(rect);

    if (markerRect !== null && rectsCollide(rect, markerRect)) {
      continue;
    }

    kept.push(annotation);
  }

  return kept;
};
