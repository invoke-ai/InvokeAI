import type { Project } from '@workbench/projectContracts';
import type { ProjectCommandResult } from '@workbench/workbenchStore';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { createDraftProject } from '@workbench/workbenchState';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { serializeProjectDocumentV3Json } from './projectDocument';

const harness = vi.hoisted(() => ({
  close: vi.fn<() => ProjectCommandResult>(),
  deleteLibraryProject: vi.fn(),
  flush: vi.fn(),
  navigate: vi.fn(),
  notifyError: vi.fn(),
  persistEmptySession: vi.fn(),
  project: {} as Project,
  releaseProjectSync: vi.fn(),
}));

vi.mock('@features/generation/react', () => ({ flushGenerateDrafts: vi.fn() }));
vi.mock('./library', () => ({
  deleteLibraryProject: harness.deleteLibraryProject,
  refreshProjectLibrary: vi.fn(),
}));
vi.mock('@tanstack/react-router', () => ({ useNavigate: () => harness.navigate }));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: harness.notifyError }) }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({ projects: { close: harness.close } }),
  useWorkbenchPersistenceAdapter: () => ({ getState: () => ({ projects: [harness.project] }) }),
  useWorkbenchPersistenceService: () => ({
    flushProjectToServer: harness.flush,
    persistEmptySession: harness.persistEmptySession,
    releaseProjectSync: harness.releaseProjectSync,
  }),
  useWorkbenchQueries: () => ({
    getProject: () => harness.project,
    getSnapshot: () => ({ projects: [harness.project, createDraftProject([harness.project])] }),
  }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { useProjectActions } from './useProjectActions';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Harness = () => {
  const { closeProject, deleteProject } = useProjectActions();
  return (
    <>
      <button onClick={() => closeProject(harness.project)}>close</button>
      <button onClick={() => void deleteProject(harness.project)}>delete</button>
    </>
  );
};

beforeEach(() => {
  accountLifecycle.activate('use-project-actions-browser-test');
  harness.project = createDraftProject([]);
  harness.close.mockReset();
  harness.close.mockReturnValue({ ok: true });
  harness.deleteLibraryProject.mockReset();
  harness.deleteLibraryProject.mockResolvedValue(undefined);
  harness.flush.mockReset();
  harness.navigate.mockReset();
  harness.notifyError.mockReset();
  harness.persistEmptySession.mockReset();
  harness.persistEmptySession.mockResolvedValue(undefined);
  harness.releaseProjectSync.mockReset();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  accountLifecycle.invalidate();
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('useProjectActions', () => {
  it('blocks project deletion while a queue run is active', async () => {
    harness.project.queue.items = [{ status: 'running' } as Project['queue']['items'][number]];
    await act(() => root?.render(<Harness />));

    await act(() => userEvent.click(document.querySelectorAll('button')[1]!));

    expect(harness.notifyError).toHaveBeenCalledWith('projects.deleteFailed', 'projects.activeRunsMustFinish');
    expect(harness.deleteLibraryProject).not.toHaveBeenCalled();
  });

  it('blocks project close before flushing while a queue run is active', async () => {
    harness.project.queue.items = [{ status: 'pending' } as Project['queue']['items'][number]];
    await act(() => root?.render(<Harness />));

    await act(() => userEvent.click(document.querySelector('button')!));

    expect(harness.notifyError).toHaveBeenCalledWith('projects.closeBlocked', 'projects.activeRunsMustFinish');
    expect(harness.flush).not.toHaveBeenCalled();
    expect(harness.close).not.toHaveBeenCalled();
  });

  it('reports a rejected close flush and keeps the tab open', async () => {
    harness.flush.mockRejectedValueOnce(new Error('network failed'));
    await act(() => root?.render(<Harness />));

    await act(() => userEvent.click(document.querySelector('button')!));
    await vi.waitFor(() => expect(harness.notifyError).toHaveBeenCalledOnce());

    expect(harness.notifyError).toHaveBeenCalledWith('projects.closeBlocked', 'network failed');
    expect(harness.close).not.toHaveBeenCalled();
  });

  it('reflushes an edit made while close is waiting for acknowledgement', async () => {
    let releaseFirst!: () => void;
    const firstReleased = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    harness.flush.mockImplementationOnce(async (project: Project) => {
      await firstReleased;
      return { documentJson: serializeProjectDocumentV3Json(project).documentJson, kind: 'acknowledged' as const };
    });
    harness.flush.mockImplementationOnce((project: Project) =>
      Promise.resolve({
        documentJson: serializeProjectDocumentV3Json(project).documentJson,
        kind: 'acknowledged' as const,
      })
    );
    await act(() => root?.render(<Harness />));

    await act(() => userEvent.click(document.querySelector('button')!));
    await vi.waitFor(() => expect(harness.flush).toHaveBeenCalledOnce());
    harness.project = { ...harness.project, name: 'edit during close' };
    releaseFirst();
    await vi.waitFor(() => expect(harness.flush).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(harness.close).toHaveBeenCalledOnce());

    expect(harness.flush.mock.calls[1]![0]).toMatchObject({ name: 'edit during close' });
    expect(harness.releaseProjectSync).toHaveBeenCalledOnce();
  });

  it('keeps the tab open when queue work starts while the close flush is in flight', async () => {
    harness.flush.mockImplementationOnce((project: Project) => {
      harness.project = {
        ...harness.project,
        queue: { items: [{ status: 'running' } as Project['queue']['items'][number]] },
      };
      return Promise.resolve({
        documentJson: serializeProjectDocumentV3Json(project).documentJson,
        kind: 'acknowledged' as const,
      });
    });
    harness.close.mockReturnValue({ ok: false, reason: 'active-queue-runs' });
    await act(() => root?.render(<Harness />));

    await act(() => userEvent.click(document.querySelector('button')!));
    await vi.waitFor(() => expect(harness.notifyError).toHaveBeenCalledOnce());

    expect(harness.notifyError).toHaveBeenCalledWith('projects.closeBlocked', 'projects.activeRunsMustFinish');
    expect(harness.close).toHaveBeenCalledOnce();
    expect(harness.releaseProjectSync).not.toHaveBeenCalled();
    expect(harness.navigate).not.toHaveBeenCalled();
  });

  it('keeps the last tab open when its empty session cannot be committed', async () => {
    harness.close.mockReturnValue({ ok: false, reason: 'last-project' });
    harness.flush.mockImplementationOnce((project: Project) =>
      Promise.resolve({
        documentJson: serializeProjectDocumentV3Json(project).documentJson,
        kind: 'acknowledged' as const,
      })
    );
    harness.persistEmptySession.mockRejectedValueOnce(new Error('session offline'));
    await act(() => root?.render(<Harness />));

    await act(() => userEvent.click(document.querySelector('button')!));
    await vi.waitFor(() => expect(harness.notifyError).toHaveBeenCalledOnce());

    expect(harness.notifyError).toHaveBeenCalledWith('projects.closeBlocked', 'session offline');
    expect(harness.releaseProjectSync).not.toHaveBeenCalled();
    expect(harness.navigate).not.toHaveBeenCalled();
  });

  it('does not navigate home when queue work starts during last-tab close', async () => {
    harness.close
      .mockReturnValueOnce({ ok: false, reason: 'last-project' })
      .mockReturnValueOnce({ ok: false, reason: 'active-queue-runs' });
    harness.flush.mockImplementationOnce((project: Project) =>
      Promise.resolve({
        documentJson: serializeProjectDocumentV3Json(project).documentJson,
        kind: 'acknowledged' as const,
      })
    );
    await act(() => root?.render(<Harness />));

    await act(() => userEvent.click(document.querySelector('button')!));
    await vi.waitFor(() => expect(harness.notifyError).toHaveBeenCalledOnce());

    expect(harness.persistEmptySession).toHaveBeenCalledOnce();
    expect(harness.notifyError).toHaveBeenCalledWith('projects.closeBlocked', 'projects.activeRunsMustFinish');
    expect(harness.releaseProjectSync).not.toHaveBeenCalled();
    expect(harness.navigate).not.toHaveBeenCalled();
  });
});
