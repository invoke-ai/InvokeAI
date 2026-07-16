export type PreviewComparisonMode = 'slider' | 'side-by-side' | 'hover';

export const DEFAULT_PREVIEW_COMPARISON_MODE: PreviewComparisonMode = 'slider';

export const getPreviewComparisonMode = (values: Record<string, unknown>): PreviewComparisonMode => {
  const mode = values.comparisonMode;

  return mode === 'side-by-side' || mode === 'hover' ? mode : DEFAULT_PREVIEW_COMPARISON_MODE;
};
