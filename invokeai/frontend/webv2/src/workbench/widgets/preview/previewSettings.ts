export type PreviewComparisonMode = 'slider' | 'side-by-side' | 'hover';

export const DEFAULT_PREVIEW_COMPARISON_MODE: PreviewComparisonMode = 'slider';

export const getPreviewComparisonMode = (values: Record<string, unknown>): PreviewComparisonMode => {
  const mode = values.comparisonMode;

  return mode === 'side-by-side' || mode === 'hover' ? mode : DEFAULT_PREVIEW_COMPARISON_MODE;
};

export const getPreviewMetadataOpen = (values: Record<string, unknown>): boolean => values.metadataOpen === true;

export const getPreviewFilmstripVisible = (values: Record<string, unknown>): boolean =>
  values.filmstripVisible !== false;
