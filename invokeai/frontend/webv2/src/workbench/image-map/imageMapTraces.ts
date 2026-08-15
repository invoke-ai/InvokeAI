import type { Layout } from 'plotly.js';

import type { ImageMapPoint } from './api';

import { getClusterColor } from './clusterPalette';

/**
 * Pure trace/layout builders for the map, kept apart from the plotly host so
 * the math is testable without WebGL. Trace identity follows PhotoMapAI:
 * named traces in a fixed z-order, with "Current Image" always last so the
 * gold marker renders on top.
 */

export const ALL_POINTS_TRACE = 'All Points';
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
  hoverinfo: 'none' | 'text';
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
 * The gold target marking the current gallery image. Built empty here; a later
 * PR restyles it live as the selection changes.
 */
export const buildCurrentImageTrace = (): ScatterTrace => ({
  customdata: [],
  hoverinfo: 'none',
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

export const buildMapLayout = (): Partial<Layout> => ({
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
    showgrid: true,
    showticklabels: false,
    zeroline: true,
    zerolinecolor: GRID_ZERO_COLOR,
    zerolinewidth: 1,
  },
});
