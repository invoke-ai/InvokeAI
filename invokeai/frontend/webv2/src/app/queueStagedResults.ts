import type { Project } from '@workbench/projectContracts';

/** Translate persisted Canvas candidate provenance into Queue's delivery-recovery read model. */
export const getQueueStagedResults = (projects: readonly Pick<Project, 'id' | 'canvas'>[]) =>
  projects.flatMap((project) =>
    project.canvas.stagingArea.pendingImages.flatMap((candidate) =>
      candidate.sourceBackendItemId === undefined
        ? []
        : [
            {
              backendItemId: candidate.sourceBackendItemId,
              projectId: project.id,
              queueItemId: candidate.sourceQueueItemId,
            },
          ]
    )
  );
