import type { FloatingWidgetState } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

import { shallowEqual, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { lazy, Suspense } from 'react';

// The window chrome loads only once a widget actually floats, keeping it out
// of the shell's eager bundle — nothing renders while no widget floats.
const FloatingWidgetWindow = lazy(() =>
  import('./FloatingWidgetWindow').then((module) => ({ default: module.FloatingWidgetWindow }))
);

const EMPTY_FLOATING: Record<WidgetInstanceId, FloatingWidgetState> = {};

/**
 * Render floating windows with z-order derived from stackOrder rank, keeping persisted ordering within UI layer
 * bounds. The windows render in a fixed order and stack by z-index alone: raising one must not move its DOM node,
 * which would drop keyboard focus from the very window that focus just raised.
 */
export const FloatingWidgetLayer = () => {
  const floatingWidgets = useActiveProjectSelector(
    (project) => project.floatingWidgets ?? EMPTY_FLOATING,
    shallowEqual
  );
  const instanceIds = Object.keys(floatingWidgets).sort();

  if (instanceIds.length === 0) {
    return null;
  }

  const stackRanks = new Map(
    [...instanceIds]
      .sort((left, right) => floatingWidgets[left].stackOrder - floatingWidgets[right].stackOrder)
      .map((instanceId, stackRank) => [instanceId, stackRank])
  );

  return (
    <Suspense fallback={null}>
      {instanceIds.map((instanceId) => (
        <FloatingWidgetWindow
          key={instanceId}
          instanceId={instanceId}
          stackRank={stackRanks.get(instanceId) ?? 0}
          state={floatingWidgets[instanceId]}
        />
      ))}
    </Suspense>
  );
};
