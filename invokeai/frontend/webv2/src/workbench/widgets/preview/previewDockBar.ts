import type { PreviewDensity } from './previewDensity';

/**
 * The bar docked under Preview's stage: the filmstrip, or the compare toolbar in its place. A full-width subtle
 * surface split from the dot-grid stage by a hairline, shared so the two read as one surface family.
 */
export const PREVIEW_DOCK_BAR_PROPS = {
  bg: 'bg.subtle',
  borderColor: 'border.subtle',
  borderTopWidth: '1px',
  flexShrink: 0,
  px: '2',
  w: 'full',
} as const;

/** The bar's height at a density; minimal Preview shows no filmstrip, so it shares compact's. */
export const getPreviewDockBarHeight = (density: PreviewDensity): string =>
  density === 'full' ? '3.75rem' : '2.75rem';
