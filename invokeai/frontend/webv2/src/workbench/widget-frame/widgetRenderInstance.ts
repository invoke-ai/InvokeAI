import type { WidgetInstanceContract } from '@workbench/widgetContracts';

export interface ProjectWidgetRenderInstance {
  instance: WidgetInstanceContract | undefined;
  projectId: string;
}

export const areWidgetRenderInstancesEqual = (
  left: WidgetInstanceContract | undefined,
  right: WidgetInstanceContract | undefined
): boolean =>
  left === right ||
  (left !== undefined &&
    right !== undefined &&
    left.id === right.id &&
    left.typeId === right.typeId &&
    left.title === right.title &&
    left.createdAt === right.createdAt);

export const areProjectWidgetRenderInstancesEqual = (
  left: ProjectWidgetRenderInstance,
  right: ProjectWidgetRenderInstance
): boolean => left.projectId === right.projectId && areWidgetRenderInstancesEqual(left.instance, right.instance);
