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
 * Renders every floated widget instance of the active project as a fixed
 * window. Mounted once in the shell; nothing renders while no widget floats.
 * z-order derives from the RANK of each window's stackOrder, not its raw
 * value — stackOrder grows monotonically forever (it is persisted), and raw
 * values would eventually climb past the UI library's layer tokens.
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
