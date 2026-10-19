export type InvokeIconMode = { mode: 'play' } | { mode: 'progress'; value: number | null };

/**
 * Replace play with progress only when running without hover or keyboard focus-visible. Preserve geometry and
 * availability; click-focus does not count.
 */
export const getInvokeIconMode = ({
  hasOpenWork,
  isHovered,
  progress,
}: {
  hasOpenWork: boolean;
  isHovered: boolean;
  progress: number | null;
}): InvokeIconMode => (hasOpenWork && !isHovered ? { mode: 'progress', value: progress } : { mode: 'play' });
