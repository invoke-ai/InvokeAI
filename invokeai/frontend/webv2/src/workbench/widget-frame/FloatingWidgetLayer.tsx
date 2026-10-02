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
 * bounds.
 */
export const FloatingWidgetLayer = () => {
  const floatingWidgets = useActiveProjectSelector(
    (project) => project.floatingWidgets ?? EMPTY_FLOATING,
    shallowEqual
  );
  const entries = Object.entries(floatingWidgets).sort(([, left], [, right]) => left.stackOrder - right.stackOrder);

  if (entries.length === 0) {
    return null;
  }

  return (
    <Suspense fallback={null}>
      {entries.map(([instanceId, state], stackRank) => (
        <FloatingWidgetWindow key={instanceId} instanceId={instanceId} stackRank={stackRank} state={state} />
      ))}
    </Suspense>
  );
};
