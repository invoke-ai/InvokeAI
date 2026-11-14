import type { Project } from '@workbench/projectContracts';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { describe, expect, it, vi } from 'vitest';

/** The hook reads and writes a real store; only the context that would hand it over is stood in for. */
const harness = vi.hoisted(() => ({
  stopGenerating: null as null | (() => void),
  store: null as ReturnType<typeof createWorkbenchStore> | null,
}));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => harness.store!.getState().activeProjectId,
  useOptionalWorkbenchCommands: () => harness.store!.commands,
  useWorkbenchCommands: () => harness.store!.commands,
  useWorkbenchQueries: () => harness.store!.queries,
}));

const { useStagingRecovery } = await import('./useCanvasRecovery');

const i18n = createInstance();
await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: {} });

/** A run still generating, bound for `project`'s canvas as it is now, or for the Gallery. */
const run = (id: string, project: Project, destination: 'canvas' | 'gallery' = 'canvas'): QueueItem =>
  ({
    backendItemIds: [1, 2],
    cancellable: true,
    id,
    snapshot: {
      canvas: { document: project.canvas.document, documentRevision: project.canvas.documentRevision },
      destination,
      presentation: { batchCount: 2, height: 64, width: 64 },
      sourceId: 'canvas',
      submittedAt: '2026-07-16T00:00:00.000Z',
    },
    status: 'running',
  }) as unknown as QueueItem;

const keepStopGenerating = (stopGenerating: () => void) => {
  harness.stopGenerating = stopGenerating;
};

const Probe = ({ onReady }: { onReady: (stopGenerating: () => void) => void }) => {
  onReady(useStagingRecovery().stopGenerating);
  return null;
};

describe('Stop generating', () => {
  it('stops every run still generating for this canvas, and nothing else, with one notice for all', async () => {
    const shown = createInitialWorkbenchState().projects[0]!;
    const other = { ...createInitialWorkbenchState().projects[0]!, id: 'other-project' };
    const initial = createInitialWorkbenchState();
    harness.store = createWorkbenchStore({
      ...initial,
      activeProjectId: shown.id,
      projects: [
        {
          ...shown,
          queue: { items: [run('canvas-1', shown), run('canvas-2', shown), run('to-gallery', shown, 'gallery')] },
        },
        { ...other, queue: { items: [run('other-canvas', other)] } },
      ],
    });
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    await act(() =>
      root.render(
        <I18nextProvider i18n={i18n}>
          <Probe onReady={keepStopGenerating} />
        </I18nextProvider>
      )
    );

    await act(() => harness.stopGenerating!());

    const statuses = Object.fromEntries(
      harness.store
        .getState()
        .projects.flatMap((project) => project.queue.items.map((item) => [item.id, item.status] as const))
    );
    expect(statuses).toEqual({
      'canvas-1': 'cancelled',
      'canvas-2': 'cancelled',
      'other-canvas': 'running',
      'to-gallery': 'running',
    });
    expect(harness.store.getState().notifications.map((notice) => notice.title)).toEqual([
      'widgets.canvas.staging.stoppedGenerating',
    ]);

    await act(() => root.unmount());
    host.remove();
  });
});
