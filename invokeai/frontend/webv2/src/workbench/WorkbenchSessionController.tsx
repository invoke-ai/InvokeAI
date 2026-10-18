import { requestLibraryWorkflowLoad } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { areArraysEqual } from '@platform/state/selectors';
import { useNavigate } from '@tanstack/react-router';
import { resolveLaunchpadIntent } from '@workbench/launchpad/intents';
import { useTranslation } from 'react-i18next';

import type { WorkbenchSearch } from './projects/session';

import { requestCanvasStartImage } from './projects/canvasStartImage';
import { takeNewProjectRequest } from './projects/newProjectRequest';
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
  const activeProjectId = useWorkbenchSelector((snapshot) => snapshot.activeProject.id);
  const { t } = useTranslation();

  useMountEffect(() => {
    if (search.new === true) {
      // Apply the draft intent before stripping it from search so it runs exactly once.
      const intent = resolveLaunchpadIntent(search.intent);

      if (intent) {
        commands.layout.applyPreset(intent.presetId);
        commands.generation.setSource(intent.sourceId);
      }

      // The dialog's choices ride alongside the URL intent; the fresh draft is the active project by now. A request
      // staged for another start is stale (a reused link, say) and must not land here.
      const staged = takeNewProjectRequest();
      const request = staged && staged.intent === intent?.id ? staged : null;

      if (request?.name) {
        commands.projects.rename(activeProjectId, request.name);
      }

      if (request?.generateSettings) {
        commands.generation.patchSettings(request.generateSettings, activeProjectId, 'system');
      }

      // The Workflow widget's pending loader takes it over the draft's untouched blank workflow.
      if (request?.libraryWorkflow) {
        requestLibraryWorkflowLoad(request.libraryWorkflow.workflowId, request.libraryWorkflow.name);
      }

      // A Canvas start sizes the canvas document to its chosen size (an image's exact dimensions, else the generate
      // size), as creating a canvas from gallery images does. The Canvas widget lays an image down once it mounts:
      // the import path is widget territory the boot graph does not carry.
      if (intent?.id === 'canvas') {
        const startImage = request?.startImage ?? null;
        const settings = request?.generateSettings;
        const canvasSize = startImage
          ? { height: startImage.height, width: startImage.width }
          : typeof settings?.width === 'number' && typeof settings.height === 'number'
            ? { height: settings.height, width: settings.width }
            : null;

        if (canvasSize) {
          const { height, width } = canvasSize;

          commands.canvas.apply(activeProjectId, { height, type: 'resizeCanvasDocument', width });
          commands.canvas.apply(activeProjectId, { bbox: { height, width, x: 0, y: 0 }, type: 'setCanvasBbox' });
        }

        if (startImage) {
          requestCanvasStartImage({ image: startImage, projectId: activeProjectId });
        }
      }

      void navigate({ replace: true, search: {}, to: '/app' });
      return;
    }

    const requestedProjectId = search.project;
    if (!requestedProjectId) {
      return;
    }

    if (projectIds.includes(requestedProjectId)) {
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
