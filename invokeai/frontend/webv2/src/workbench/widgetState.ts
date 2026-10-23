import type { Project } from '@workbench/projectContracts';
import type { WidgetInstanceContract, WidgetStateContract, WidgetTypeId } from '@workbench/widgetContracts';

const emptyWidgetState = (widgetId: WidgetTypeId): WidgetStateContract => ({
  id: widgetId,
  label: widgetId,
  values: {},
  version: 1,
});

/** The widget instances alone: readers of a widget's values need nothing else from the project. */
export type WidgetInstanceHolder = Pick<Project, 'widgetInstances'>;

export const getProjectWidgetInstance = (
  project: WidgetInstanceHolder,
  widgetId: WidgetTypeId
): WidgetInstanceContract | undefined =>
  project.widgetInstances[widgetId] ??
  Object.values(project.widgetInstances).find((instance) => instance.typeId === widgetId);

export const getProjectWidgetState = (project: WidgetInstanceHolder, widgetId: WidgetTypeId): WidgetStateContract =>
  getProjectWidgetInstance(project, widgetId)?.state ?? emptyWidgetState(widgetId);

export const getProjectWidgetValues = (
  project: WidgetInstanceHolder,
  widgetId: WidgetTypeId
): Record<string, unknown> => getProjectWidgetState(project, widgetId).values;
