import type { Project, WidgetInstanceContract, WidgetStateContract, WidgetTypeId } from './types';

const emptyWidgetState = (widgetId: WidgetTypeId): WidgetStateContract => ({
  id: widgetId,
  label: widgetId,
  values: {},
  version: 1,
});

export const getProjectWidgetInstance = (
  project: Project,
  widgetId: WidgetTypeId
): WidgetInstanceContract | undefined =>
  project.widgetInstances[widgetId] ??
  Object.values(project.widgetInstances).find((instance) => instance.typeId === widgetId);

export const getProjectWidgetState = (project: Project, widgetId: WidgetTypeId): WidgetStateContract =>
  getProjectWidgetInstance(project, widgetId)?.state ?? emptyWidgetState(widgetId);

export const getProjectWidgetValues = (project: Project, widgetId: WidgetTypeId): Record<string, unknown> =>
  getProjectWidgetState(project, widgetId).values;
