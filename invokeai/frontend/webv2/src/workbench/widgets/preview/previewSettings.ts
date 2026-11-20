import type { ResolvedValues } from '@platform/core/resolvedValues';

export type PreviewComparisonMode = 'slider' | 'side-by-side' | 'hover';

export const DEFAULT_PREVIEW_COMPARISON_MODE: PreviewComparisonMode = 'slider';

export const getPreviewComparisonMode = (values: ResolvedValues): PreviewComparisonMode => {
  const mode = values.comparisonMode;

  return mode === 'side-by-side' || mode === 'hover' ? mode : DEFAULT_PREVIEW_COMPARISON_MODE;
};

export const getPreviewMetadataOpen = (values: Record<string, unknown>): boolean => values.metadataOpen === true;

export const getPreviewFilmstripVisible = (values: ResolvedValues): boolean => values.filmstripVisible !== false;

/** Whether the Preview's videos loop: on until the viewer turns it off in the player. */
export const getPreviewVideoLoop = (values: Record<string, unknown>): boolean => values.videoLoop !== false;
