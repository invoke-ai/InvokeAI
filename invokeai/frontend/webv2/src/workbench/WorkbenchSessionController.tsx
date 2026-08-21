import { flushGenerateDrafts } from '@features/generation/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { areArraysEqual } from '@platform/state/selectors';
import { useNavigate } from '@tanstack/react-router';
import { resolveLaunchpadIntent } from '@workbench/launchpad/intents';

import type { WorkbenchSearch } from './projects/session';

import {
  useWorkbenchCommands,
  useWorkbenchHasHydrated,
  useWorkbenchPersistenceService,
  useWorkbenchSelector,
} from './WorkbenchContext';

const HydratedSessionController = ({ search }: { search: WorkbenchSearch }) => {
  const commands = useWorkbenchCommands();
  const navigate = useNavigate();
  const persistence = useWorkbenchPersistenceService();
  const projectIds = useWorkbenchSelector((snapshot) => snapshot.projects.map((project) => project.id), areArraysEqual);

  useMountEffect(() => {
    if (search.new === true) {
      // Arrange the fresh draft before the search is stripped, so the intent
      // applies exactly once and never survives into a later navigation.
      const intent = resolveLaunchpadIntent(search.intent);

      // A named preset wins: the user picked the arrangement itself, so it
      // outranks whatever arrangement an intent merely implies. The intent
      // still gets to name the invocation source, which a preset cannot.
      if (search.preset) {
        commands.layout.applyPreset(search.preset);
      } else if (intent) {
        commands.layout.applyPreset(intent.presetId);
      }

      if (intent) {
        commands.generation.setSource(intent.sourceId);
      }

      void navigate({ replace: true, search: {}, to: '/app' });
      return;
    }

    const requestedProjectId = search.project;
    if (!requestedProjectId) {
      return;
    }

    if (projectIds.includes(requestedProjectId)) {
      flushGenerateDrafts();
      commands.projects.switchTo(requestedProjectId);
      return;
    }

    let isCancelled = false;
    void persistence.hydrateProjectFromServer(requestedProjectId).then((project) => {
      if (isCancelled) {
        return;
      }

      if (project) {
        flushGenerateDrafts();
        commands.projects.open(project);
      } else {
        commands.notifications.add({
          kind: 'info',
          message: 'The linked project does not exist on this account — it may have been deleted.',
          title: 'Project not found',
        });
      }
    });

    return () => {
      isCancelled = true;
    };
  });

  return null;
};

/**
 * Search changes remount the lifecycle adapter, eliminating effect-based prop
 * synchronization while preserving cancellation for asynchronous hydration.
 */
export const WorkbenchSessionController = ({ search }: { search: WorkbenchSearch }) => {
  const hasHydrated = useWorkbenchHasHydrated();

  if (!hasHydrated) {
    return null;
  }

  return (
    <HydratedSessionController
      key={`${search.new === true}:${search.project ?? ''}:${search.intent ?? ''}`}
      search={search}
    />
  );
};
