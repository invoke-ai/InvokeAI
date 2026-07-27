export const LAUNCHPAD_READY_MARK = 'invokeai:ready:launchpad';

export const getWidgetReadyMark = (region: string, typeId: string): string =>
  `invokeai:ready:widget:${region}:${typeId}`;

/**
 * Records a product-owned readiness milestone for the browser performance
 * harness. Keeping only the latest mark makes repeated layout/project
 * transitions unambiguous while preserving the mark's navigation-relative
 * start time.
 */
export const markSemanticReady = (name: string): void => {
  if (typeof performance === 'undefined' || typeof performance.mark !== 'function') {
    return;
  }

  performance.clearMarks?.(name);
  performance.mark(name);
};
