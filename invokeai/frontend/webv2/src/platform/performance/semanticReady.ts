export const LAUNCHPAD_READY_MARK = 'invokeai:ready:launchpad';

export const getWidgetReadyMark = (region: string, typeId: string): string =>
  `invokeai:ready:widget:${region}:${typeId}`;

/** Keep only the latest readiness mark per milestone while retaining navigation-relative timing. */
export const markSemanticReady = (name: string): void => {
  if (typeof performance === 'undefined' || typeof performance.mark !== 'function') {
    return;
  }

  performance.clearMarks?.(name);
  performance.mark(name);
};
