import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchQueries, useWorkbenchSubscription } from '@workbench/WorkbenchContext';

import { readCanvasScrollInput } from './canvasSettings';

// Per browser, not per account or project: the pointing device belongs to the machine.
const TRACKPAD_DETECTED_KEY = 'invokeai:v7:webv2:canvas-trackpad-detected';

const readTrackpadDetected = (): boolean => {
  try {
    return globalThis.localStorage?.getItem(TRACKPAD_DETECTED_KEY) === 'true';
  } catch {
    return false;
  }
};

const persistTrackpadDetected = (): void => {
  try {
    globalThis.localStorage?.setItem(TRACKPAD_DETECTED_KEY, 'true');
  } catch {
    // Without storage the next session detects the trackpad again.
  }
};

/** Feeds the scroll input setting to the engine and keeps the engine's trackpad detection across sessions. */
export const CanvasScrollInputFeed = ({
  engine,
  projectId,
}: {
  engine: Pick<CanvasEngineHandle, 'interaction'>;
  projectId: string;
}) => {
  const queries = useWorkbenchQueries();
  const subscribe = useWorkbenchSubscription();
  useMountEffect(() => {
    const { interaction } = engine;
    const feed = (): void => {
      const project = queries.getProject(projectId);
      if (project) {
        const values = getResolvedProjectWidgetValues(project, 'canvas', queries.getSnapshot().scopedDefaults);
        interaction.set('scrollInput', readCanvasScrollInput(values));
      }
    };
    feed();
    if (readTrackpadDetected()) {
      interaction.set('trackpadDetected', true);
    }
    const unsubscribeDetected = interaction.subscribe('trackpadDetected', () => {
      if (interaction.get('trackpadDetected')) {
        persistTrackpadDetected();
      }
    });
    const unsubscribeWorkbench = subscribe(feed);
    return () => {
      unsubscribeWorkbench();
      unsubscribeDetected();
    };
  });
  return null;
};
