import type { WidgetContributionSource } from '@workbench/widgetContracts';

const GLOBAL_SOURCE_KEY = 'global';

export const areWidgetContributionSourcesEqual = (
  left: WidgetContributionSource | null | undefined,
  right: WidgetContributionSource | null | undefined
): boolean => {
  if (!left || !right) {
    return false;
  }

  return (
    left.projectId === right.projectId &&
    left.region === right.region &&
    left.typeId === right.typeId &&
    left.instanceId === right.instanceId
  );
};

export const getWidgetContributionSourceKey = (source: WidgetContributionSource | null | undefined): string =>
  source
    ? JSON.stringify(['widget', source.projectId, source.region, source.typeId, source.instanceId])
    : GLOBAL_SOURCE_KEY;

export const getWidgetContributionRegistrationKey = (
  id: string,
  source: WidgetContributionSource | null | undefined
): string => JSON.stringify([id, getWidgetContributionSourceKey(source)]);
