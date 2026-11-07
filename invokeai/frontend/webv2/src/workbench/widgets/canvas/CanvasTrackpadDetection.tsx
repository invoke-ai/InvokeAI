import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { WorkbenchQueries } from '@workbench/workbenchStore';

import { useMountEffect } from '@platform/react/useMountEffect';
import { hasHydratedWorkbenchPreferences, useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { getProjectWidgetValues, getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchQueries } from '@workbench/WorkbenchContext';

import { CANVAS_TRACKPAD_NAVIGATION_KEY } from './canvasSettings';
import { autoEnableTrackpadNavigation, isTrackpadNavigationUndecided } from './trackpadNavigation';

type TrackpadEngine = Pick<CanvasEngineHandle, 'interaction' | 'tools'>;

/**
 * Subscribes trackpad auto-enable (see `trackpadNavigation.ts`) to the engine's pinch reports. The engine store is set
 * at once, before the pinch is routed, so the pinch that triggered it already zooms; the saved default then feeds the
 * same value. A project that pins the setting is left alone. If the saved preferences say it was decided elsewhere,
 * the store goes back to the project's resolved value, even after unsubscribing: adopting that decision disarms
 * detection and unmounts the subscriber, while the settings feed sees no change to re-apply. Setting a store of a
 * disposed engine is harmless. Returns the unsubscribe.
 */
export const subscribeTrackpadDetection = (
  engine: TrackpadEngine,
  queries: Pick<WorkbenchQueries, 'getProject' | 'getSnapshot'>,
  projectId: string
): (() => void) => {
  return engine.tools.onTrackpadPinch(() => {
    if (!isTrackpadNavigationUndecided()) {
      return;
    }
    const project = queries.getProject(projectId);
    if (!project || Object.hasOwn(getProjectWidgetValues(project, 'canvas'), CANVAS_TRACKPAD_NAVIGATION_KEY)) {
      return;
    }
    engine.interaction.set('trackpadNavigation', true);
    void autoEnableTrackpadNavigation().then((enabled) => {
      const current = queries.getProject(projectId);
      if (enabled || !current) {
        return;
      }
      const values = getResolvedProjectWidgetValues(current, 'canvas', queries.getSnapshot().scopedDefaults);
      engine.interaction.set('trackpadNavigation', values[CANVAS_TRACKPAD_NAVIGATION_KEY] === true);
    });
  });
};

/**
 * Whether pinch detection should listen: a pinch could still turn the setting on (preferences loaded, not decided, the
 * project does not pin it). While it listens the engine swallows pinch-shaped ctrl+wheel instead of sizing the brush,
 * so once decided (an explicit off included) the listener goes and pinches size the brush again.
 */
export const useTrackpadDetectionArmed = (): boolean => {
  const undecided = useWorkbenchSettingsSelector(
    (snapshot) => hasHydratedWorkbenchPreferences(snapshot) && !snapshot.preferences.canvasTrackpadNavigationDecided
  );
  const pinned = useActiveProjectSelector((project) =>
    Object.hasOwn(getProjectWidgetValues(project, 'canvas'), CANVAS_TRACKPAD_NAVIGATION_KEY)
  );
  return undecided && !pinned;
};

export const CanvasTrackpadDetection = ({ engine, projectId }: { engine: TrackpadEngine; projectId: string }) => {
  const queries = useWorkbenchQueries();
  useMountEffect(() => subscribeTrackpadDetection(engine, queries, projectId));
  return null;
};
