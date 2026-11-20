export type InvokeIconMode = { mode: 'front' } | { mode: 'play' } | { mode: 'progress'; value: number | null };

/** Replace play with progress only when running without hover or keyboard focus-visible. */
export const getInvokeIconMode = ({
  hasOpenWork,
  isFrontHeld,
  isHovered,
  progress,
}: {
  hasOpenWork: boolean;
  /** Shift is held and a click would queue: callers pass false while Invoke cannot queue. */
  isFrontHeld: boolean;
  isHovered: boolean;
  progress: number | null;
}): InvokeIconMode => {
  if (hasOpenWork && !isHovered) {
    return { mode: 'progress', value: progress };
  }

  return isHovered && isFrontHeld ? { mode: 'front' } : { mode: 'play' };
};
