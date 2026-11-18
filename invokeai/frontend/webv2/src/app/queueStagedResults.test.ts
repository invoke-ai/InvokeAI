import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { expect, it } from 'vitest';

import { getQueueStagedResults } from './queueStagedResults';

it('projects real reducer candidates, including multiple outputs, while excluding absent backend provenance', () => {
  let state = createInitialWorkbenchState();
  const projectId = state.activeProjectId;
  state = workbenchReducer(state, {
    type: 'stageRetriedCanvasResults',
    backendItemId: 99,
    projectId,
    queueItemId: 'run',
    images: ['one.png', 'two.png'].map((imageName) => ({
      imageName,
      sourceQueueItemId: 'run',
      queuedAt: '2026-11-05T00:00:00Z',
      width: 64,
      height: 64,
      imageUrl: `/${imageName}`,
      thumbnailUrl: `/${imageName}`,
    })),
  });
  const project = state.projects.find((project) => project.id === projectId)!;
  expect(project.canvas.stagingArea.pendingImages).toHaveLength(2);
  const unknown = { ...project.canvas.stagingArea.pendingImages[0]!, id: 'unknown', sourceBackendItemId: undefined };
  const projects = [
    {
      ...project,
      canvas: {
        ...project.canvas,
        stagingArea: {
          ...project.canvas.stagingArea,
          pendingImages: [...project.canvas.stagingArea.pendingImages, unknown],
        },
      },
    },
  ];
  expect(getQueueStagedResults(projects)).toEqual([
    { backendItemId: 99, projectId, queueItemId: 'run' },
    { backendItemId: 99, projectId, queueItemId: 'run' },
  ]);
});
