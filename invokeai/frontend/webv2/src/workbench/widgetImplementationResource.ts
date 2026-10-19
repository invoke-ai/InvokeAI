import type { WidgetImplementation, WidgetImplementationResource, WidgetTypeId } from './widgetContracts';

import { createDeferredResource } from './deferredResource';

const validateImplementation = (widgetId: WidgetTypeId, value: WidgetImplementation): WidgetImplementation => {
  if (!value || typeof value !== 'object' || typeof value.view !== 'function') {
    throw new TypeError(`Widget ${widgetId} implementation must provide a view component.`);
  }

  return value;
};

export const createWidgetImplementationResource = (
  widgetId: WidgetTypeId,
  loader: () => Promise<WidgetImplementation>
): WidgetImplementationResource =>
  createDeferredResource(loader, (implementation) => validateImplementation(widgetId, implementation));
