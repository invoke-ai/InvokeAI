import type { ProjectWorkflowSource } from '@features/workflow/contracts';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchQueueItem } from '@workbench/queueHistoryContracts';

import { createDefaultVideoWidgetValues } from '@features/video';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { createBlankWorkflowDocument, createProjectWorkflowCollection } from '@workbench/projectWorkflows';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ apiFetch: vi.fn() }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  apiFetch: mocks.apiFetch,
}));

import {
  getCompletedLibraryWorkflowRun,
  withLibraryWorkflowRunTouch,
  withQueueRecallCache,
} from './queueRunPersistence';

it('retains exact recall when a terminal item disappears before journal settlement', async () => {
  const put = vi.fn().mockResolvedValue({ kind: 'stored' });
  const settle = vi.fn().mockResolvedValue({ kind: 'removed' });
  const journal = { record: vi.fn().mockResolvedValue({ kind: 'stored' }), settle };
  const videoValues = createDefaultVideoWidgetValues();
  const item = {
    id: 'q',
    status: 'pending',
    snapshot: {
      sourceId: 'video',
      submittedAt: '2026-09-04T00:00:00Z',
      recall: { videoValues },
    },
  } as WorkbenchQueueItem;
  const persistence = withQueueRecallCache(journal, { put }, () => undefined);
  await persistence.record('p', item);
  await persistence.settle('p', 'q');
  expect(put).toHaveBeenCalledWith({
    projectId: 'p',
    queueItemId: 'q',
    sourceId: 'video',
    submittedAt: '2026-09-04T00:00:00Z',
    videoValues,
  });
  expect(settle).toHaveBeenCalledWith('p', 'q');
});

it('does not prevent run cleanup when the optional recall cache fails', async () => {
  const settle = vi.fn().mockResolvedValue({ kind: 'removed' });
  const item = {
    id: 'q',
    status: 'failed',
    snapshot: {
      sourceId: 'video',
      submittedAt: '2026-09-04T00:00:00Z',
      recall: { videoValues: createDefaultVideoWidgetValues() },
    },
  } as WorkbenchQueueItem;
  const persistence = withQueueRecallCache(
    { record: vi.fn(), settle },
    { put: vi.fn().mockRejectedValue(new Error('quota')) },
    () => item
  );
  await expect(persistence.settle('p', 'q')).resolves.toEqual({ kind: 'removed' });
});

type RunProject = Pick<Project, 'queue' | 'workflows'>;

const document = createBlankWorkflowDocument();

const project = (
  status: WorkbenchQueueItem['status'],
  { projectWorkflowId = document.id, source }: { projectWorkflowId?: string; source?: ProjectWorkflowSource } = {}
): RunProject => {
  const workflows = createProjectWorkflowCollection(document);
  const queueItem = {
    cancellable: false,
    id: 'run-1',
    snapshot: { backendSubmission: { batchCount: 1, graph: {}, kind: 'workflow', projectWorkflowId } },
    status,
  } as unknown as WorkbenchQueueItem;

  return {
    queue: { items: [queueItem] },
    workflows: {
      ...workflows,
      entries: workflows.entries.map((entry) => ({ ...entry, ...(source ? { source } : {}) })),
    },
  };
};

const linked = { source: { libraryWorkflowId: 'wf-user-1', ownerUserId: 'user-a', revision: 3 } };

describe('getCompletedLibraryWorkflowRun', () => {
  const decide = (before: RunProject, after: RunProject, viewerUserId: string | null = null) =>
    getCompletedLibraryWorkflowRun(before, after, 'run-1', viewerUserId);

  it('names the linked template when a run completes', () => {
    expect(decide(project('running', linked), project('completed', linked))).toBe('wf-user-1');
    expect(decide(project('running', linked), project('completed', linked), 'user-a')).toBe('wf-user-1');
  });

  it('bumps a completion only once', () => {
    expect(decide(project('completed', linked), project('completed', linked))).toBe(null);
  });

  it('leaves unlinked workflows and templates another account owns alone', () => {
    expect(decide(project('running'), project('completed'))).toBe(null);
    expect(decide(project('running', linked), project('completed', linked), 'user-b')).toBe(null);
  });

  it('ignores runs that did not complete or whose workflow is gone', () => {
    expect(decide(project('running', linked), project('cancelled', linked))).toBe(null);
    expect(
      decide(project('running', linked), project('completed', { ...linked, projectWorkflowId: 'deleted-workflow' }))
    ).toBe(null);
  });
});

describe('withLibraryWorkflowRunTouch', () => {
  const payload = { projectId: 'project-1', queueItemId: 'run-1' };
  let current: RunProject;
  let disposed: boolean;

  const setup = (source: ProjectWorkflowSource = linked.source) => {
    current = project('running', { source });
    disposed = false;
    const routeResults = vi.fn(() => {
      current = project('completed', { source });
    });

    return {
      route: withLibraryWorkflowRunTouch(routeResults, {
        getProject: (projectId) => (projectId === payload.projectId ? current : undefined),
        isDisposed: () => disposed,
        owner: captureAccountScope(),
        viewerUserId: null,
      }),
      routeResults,
    };
  };

  /** The workflow data module loads lazily; let the import and the request it starts settle. */
  const settle = async () => {
    await vi.dynamicImportSettled();
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  };

  beforeEach(() => {
    mocks.apiFetch.mockReset().mockResolvedValue(new Response(null, { status: 204 }));
    accountLifecycle.activate('user-a');
  });

  it('marks the template run once per completion', async () => {
    const { route, routeResults } = setup();

    route(payload);
    route(payload);
    await settle();

    expect(routeResults).toHaveBeenCalledTimes(2);
    expect(mocks.apiFetch).toHaveBeenCalledTimes(1);
    expect(mocks.apiFetch).toHaveBeenCalledWith(
      '/api/v1/workflows/i/wf-user-1/last_run_at',
      expect.objectContaining({ method: 'PUT' })
    );
  });

  it('never marks a bundled template', async () => {
    setup({ libraryWorkflowId: 'default_text_to_image', revision: null }).route(payload);
    await settle();

    expect(mocks.apiFetch).not.toHaveBeenCalled();
  });

  it('sends nothing once the runtime is disposed or the account changed, even while the module loads', async () => {
    const afterDispose = setup();
    disposed = true;
    afterDispose.route(payload);

    const afterAccountChange = setup();
    accountLifecycle.activate('user-b');
    afterAccountChange.route(payload);

    const changedWhileLoading = setup();
    changedWhileLoading.route(payload);
    accountLifecycle.activate('user-c');
    await settle();

    expect(afterDispose.routeResults).toHaveBeenCalledTimes(1);
    expect(afterAccountChange.routeResults).toHaveBeenCalledTimes(1);
    expect(changedWhileLoading.routeResults).toHaveBeenCalledTimes(1);
    expect(mocks.apiFetch).not.toHaveBeenCalled();
  });

  it('keeps a failed request and a failing project read out of result routing', async () => {
    mocks.apiFetch.mockRejectedValue(new Error('offline'));
    const { route, routeResults } = setup();

    expect(() => route(payload)).not.toThrow();
    await settle();

    const throwing = withLibraryWorkflowRunTouch(routeResults, {
      getProject: () => {
        throw new Error('snapshot unavailable');
      },
      isDisposed: () => false,
      owner: captureAccountScope(),
      viewerUserId: null,
    });

    expect(() => throwing(payload)).not.toThrow();
    await settle();
    expect(routeResults).toHaveBeenCalledTimes(2);
    expect(mocks.apiFetch).toHaveBeenCalledTimes(1);
  });
});
