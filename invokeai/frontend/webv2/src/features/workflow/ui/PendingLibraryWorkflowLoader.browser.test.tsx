/* eslint-disable react-perf/jsx-no-new-object-as-prop -- the test wires a fake adapter directly */
import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { requestLibraryWorkflowLoad, workflowUiStore } from '@features/workflow/ui/workflowUiStore';
import { createProjectGraph } from '@features/workflow/utility';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PendingWorkflowLoader } from './PendingLibraryWorkflowLoader';

const api = vi.hoisted(() => ({ getLibraryWorkflowRecord: vi.fn() }));

vi.mock('@features/workflow/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLibraryWorkflowRecord: api.getLibraryWorkflowRecord,
  touchLibraryWorkflowOpenedAt: () => Promise.resolve(),
}));

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const copyOf = (libraryWorkflowId: string): ProjectWorkflowEntry => ({
  document: { ...createProjectGraph('copy-1'), name: 'My tweaks' },
  source: { libraryWorkflowId, revision: 1 },
});

describe('PendingWorkflowLoader', () => {
  let host: HTMLDivElement;
  let root: Root;
  let addWorkflow: ReturnType<typeof vi.fn>;

  const render = (workflows: ProjectWorkflowEntry[]) =>
    act(() =>
      root.render(
        <WorkflowUiProvider
          adapter={
            {
              commands: { addWorkflow },
              notifications: { error: vi.fn(), info: vi.fn() },
              project: { getSnapshot: () => ({ id: 'project-1', workflows }), subscribe: () => () => {} },
            } as unknown as WorkflowUiAdapter
          }
        >
          <PendingWorkflowLoader />
        </WorkflowUiProvider>
      )
    );

  beforeEach(() => {
    addWorkflow = vi.fn();
    api.getLibraryWorkflowRecord.mockReset().mockResolvedValue({
      name: 'Portrait',
      revision: 2,
      workflow: { edges: [], name: 'Portrait', nodes: [] },
      workflow_id: 'lib-portrait',
    });
    workflowUiStore.patchSnapshot({ libraryCopyChoice: null, pendingWorkflowLoad: null });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  // The palette once resumed the first copy silently; it now asks, as the library does.
  it('asks about a template the project already holds, under the template’s name', async () => {
    await render([copyOf('lib-portrait')]);

    await act(() => requestLibraryWorkflowLoad('lib-portrait', 'Portrait'));

    await vi.waitFor(() =>
      expect(workflowUiStore.getSnapshot().libraryCopyChoice).toMatchObject({
        item: { name: 'Portrait', workflow_id: 'lib-portrait' },
        projectId: 'project-1',
      })
    );
    expect(api.getLibraryWorkflowRecord).not.toHaveBeenCalled();
    expect(addWorkflow).not.toHaveBeenCalled();
  });

  it('adds the first copy of a template the project does not hold', async () => {
    await render([copyOf('lib-other')]);

    await act(() => requestLibraryWorkflowLoad('lib-portrait', 'Portrait'));

    await vi.waitFor(() => expect(addWorkflow).toHaveBeenCalledTimes(1));
    expect(addWorkflow.mock.calls[0]?.[1]).toMatchObject({
      reusePlaceholder: true,
      source: { libraryWorkflowId: 'lib-portrait', revision: 2 },
    });
    expect(workflowUiStore.getSnapshot().libraryCopyChoice).toBeNull();
  });
});
