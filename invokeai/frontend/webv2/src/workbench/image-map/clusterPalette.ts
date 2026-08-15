/**
 * Categorical colors for map clusters, cycled by cluster id. Modeled on
 * PhotoMapAI's cluster palette: saturated, well-separated hues that read on
 * both light and dark plot backgrounds.
 */
export const CLUSTER_PALETTE = [
  '#4E79A7',
  '#F28E2B',
  '#E15759',
  '#76B7B2',
  '#59A14F',
  '#EDC948',
  '#B07AA1',
  '#FF9DA7',
  '#9C755F',
  '#BAB0AC',
  '#86BCB6',
  '#D37295',
  '#FABFD2',
  '#B6992D',
  '#499894',
] as const;

/** Color for DBSCAN noise (cluster -1): a neutral gray, drawn dimmed. */
export const NOISE_COLOR = '#8A8A8A';

export const getClusterColor = (cluster: number): string => {
  if (cluster < 0) {
    return NOISE_COLOR;
  }

  return CLUSTER_PALETTE[cluster % CLUSTER_PALETTE.length];
};
