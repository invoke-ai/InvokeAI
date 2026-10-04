import { flushGenerateDrafts } from '@features/generation/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { areArraysEqual } from '@platform/state/selectors';
import { useNavigate } from '@tanstack/react-router';
import { resolveLaunchpadIntent } from '@workbench/launchpad/intents';
import { useTranslation } from 'react-i18next';

import type { WorkbenchSearch } from './projects/session';

import { describeRefusedProject } from './projects/projectLoadRefusal';
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
  const { t } = useTranslation();

  useMountEffect(() => {
    if (search.new === true) {
      // Apply the draft intent before stripping it from search so it runs exactly once.
      const intent = resolveLaunchpadIntent(search.intent);

      if (intent) {
        commands.layout.applyPreset(intent.presetId);
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
      // The link is a one-time open request. Keeping it would override later
      // project selections (including a new project) on reload.
      void navigate({ replace: true, search: {}, to: '/app' });
      return;
    }

    let isCancelled = false;
    void persistence.hydrateProjectFromServer(requestedProjectId).then((result) => {
      if (isCancelled) {
        return;
      }

      if (result.status === 'loaded') {
        flushGenerateDrafts();
        commands.projects.open(result.project);
        void navigate({ replace: true, search: {}, to: '/app' });
      } else if (result.status === 'refused') {
        commands.notifications.add({ kind: 'error', ...describeRefusedProject(result.refused, t) });
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

/** Key search-dependent hydration lifecycles to cancel stale asynchronous loads. */
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
