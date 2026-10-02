import type { Project } from '@workbench/projectContracts';
import type { ProjectCommandResult } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createDraftProject } from '@workbench/workbenchState';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const harness = vi.hoisted(() => ({
  abortProjectResolution: vi.fn(),
  acknowledgeProjectResolution: vi.fn(),
  closeProject: vi.fn<() => ProjectCommandResult>(() => ({ ok: true })),
  conflict: undefined as
    | { detectedAt: string; kind: 'deleted' }
    | { detectedAt: string; kind: 'revision'; serverRevision: number }
    | undefined,
  createProject: vi.fn(),
  deleteRecoverableDraft: vi.fn(() => Promise.resolve()),
  downloadText: vi.fn(),
  getRecoverableDraftDocument: vi.fn(() => Promise.resolve('{"documentSchemaVersion":3}')),
  project: {} as Project,
  replaceProjectFromServer: vi.fn(),
  recoverableDrafts: [] as Array<{
    editorSessionId: string;
    generation: number;
    projectId: string;
    updatedAt: number;
  }>,
  resolveConflictSaveAsNew: vi.fn(() =>
    Promise.resolve({
      boardId: 'board-copy',
      name: 'Project (copy)',
      project: { id: 'copy', name: 'Project (copy)' } as Project,
      sourceName: 'Project',
      sourceProjectId: 'project-1',
      targetProjectId: 'copy',
    })
  ),
  resolveConflictDiscard: vi.fn(() => Promise.resolve()),
  resolveConflictUseServer: vi.fn(),
  retargetProject: vi.fn<() => ProjectCommandResult>(() => ({ ok: true })),
  reportError: vi.fn(),
}));

vi.mock('@platform/browser/downloadBlob', () => ({ downloadText: harness.downloadText }));
vi.mock('@platform/ui/ConfirmDialog', () => ({
  ConfirmDialog: ({ isOpen, onConfirm }: { isOpen: boolean; onConfirm(): Promise<void> | void }) =>
    isOpen ? (
      <button data-testid="confirm-resolution" onClick={() => void onConfirm()}>
        confirm
      </button>
    ) : null,
}));
vi.mock('@workbench/projects/syncStore', () => ({
  useProjectSyncSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({
      localDraftStatus: 'ok',
      projects: { 'project-1': { conflict: harness.conflict, isPendingPush: true, revision: 1 } },
      recoverableDrafts: harness.recoverableDrafts,
    }),
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProject: () => harness.project,
  useActiveProjectId: () => 'project-1',
  useActiveProjectName: () => 'Project',
  useWorkbenchCommands: () => ({
    notifications: { reportError: harness.reportError },
    projects: { close: harness.closeProject, create: harness.createProject },
  }),
  useWorkbenchPersistenceAdapter: () => ({
    replaceProjectFromServer: harness.replaceProjectFromServer,
    retargetProject: harness.retargetProject,
  }),
  useWorkbenchPersistenceService: () => ({
    acknowledgeProjectResolution: harness.acknowledgeProjectResolution,
    abortProjectResolution: harness.abortProjectResolution,
    deleteRecoverableDraft: harness.deleteRecoverableDraft,
    getRecoverableDraftDocument: harness.getRecoverableDraftDocument,
    getProjectDraftDocument: vi.fn(),
    resolveConflictDiscard: harness.resolveConflictDiscard,
    resolveConflictSaveAsNew: harness.resolveConflictSaveAsNew,
    resolveConflictUseServer: harness.resolveConflictUseServer,
  }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { ProjectConflictBanner } from './ProjectConflictBanner';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  harness.project = { ...createDraftProject([]), id: 'project-1', name: 'Project' };
  harness.conflict = undefined;
  harness.recoverableDrafts = [];
  harness.downloadText.mockClear();
  harness.deleteRecoverableDraft.mockClear();
  harness.getRecoverableDraftDocument.mockClear();
  harness.resolveConflictSaveAsNew.mockClear();
  harness.retargetProject.mockClear();
  harness.retargetProject.mockReturnValue({ ok: true });
  harness.acknowledgeProjectResolution.mockClear();
  harness.abortProjectResolution.mockClear();
  harness.closeProject.mockClear();
  harness.closeProject.mockReturnValue({ ok: true });
  harness.createProject.mockClear();
  harness.replaceProjectFromServer.mockClear();
  harness.reportError.mockClear();
  harness.resolveConflictDiscard.mockClear();
  harness.resolveConflictDiscard.mockResolvedValue(undefined);
  harness.resolveConflictUseServer.mockReset();
  harness.resolveConflictUseServer.mockResolvedValue({ project: harness.project, status: 'loaded' });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ProjectConflictBanner', () => {
  it('makes save-as-new reachable and hands the acknowledged identity to the reducer', async () => {
    harness.conflict = { detectedAt: '2026-09-03T12:00:00.000Z', kind: 'revision', serverRevision: 2 };
    await renderBanner();
    const button = Array.from(document.querySelectorAll('button')).find(
      (candidate) => candidate.textContent === 'shell.projectConflict.saveAsNew'
    );
    expect(button).toBeDefined();

    await act(() => userEvent.click(button!));
    const confirm = document.querySelector<HTMLButtonElement>('[data-testid="confirm-resolution"]');
    expect(confirm).not.toBeNull();
    await act(() => userEvent.click(confirm!));
    await vi.waitFor(() => expect(harness.retargetProject).toHaveBeenCalledOnce());

    expect(harness.resolveConflictSaveAsNew).toHaveBeenCalledWith(harness.project);
    expect(harness.retargetProject).toHaveBeenCalledWith(
      expect.objectContaining({ projectId: 'project-1', targetProjectId: 'copy' })
    );
  });

  it('releases the save-as-new fence when the reducer cannot retarget', async () => {
    harness.conflict = { detectedAt: '2026-09-03T12:00:00.000Z', kind: 'revision', serverRevision: 2 };
    harness.retargetProject.mockReturnValueOnce({ ok: false, reason: 'target-already-open' });
    await renderBanner();
    const button = Array.from(document.querySelectorAll('button')).find(
      (candidate) => candidate.textContent === 'shell.projectConflict.saveAsNew'
    );

    await act(() => userEvent.click(button!));
    await act(() => userEvent.click(document.querySelector<HTMLButtonElement>('[data-testid="confirm-resolution"]')!));
    await vi.waitFor(() => expect(harness.abortProjectResolution).toHaveBeenCalledWith('project-1'));

    expect(harness.acknowledgeProjectResolution).not.toHaveBeenCalled();
    expect(harness.reportError).toHaveBeenCalledOnce();
  });

  it('exports the live project including edits newer than the autosave draft', async () => {
    harness.conflict = { detectedAt: '2026-09-03T12:00:00.000Z', kind: 'revision', serverRevision: 2 };
    harness.project = { ...harness.project, name: 'unsaved click-time edit' };
    await renderBanner();
    const button = Array.from(document.querySelectorAll('button')).find(
      (candidate) => candidate.textContent === 'shell.projectConflict.export'
    );

    await act(() => userEvent.click(button!));
    await vi.waitFor(() => expect(harness.downloadText).toHaveBeenCalledOnce());

    expect(JSON.parse(harness.downloadText.mock.calls[0]![0])).toMatchObject({ name: 'unsaved click-time edit' });
  });

  it('replaces the reducer project before releasing a use-server fence', async () => {
    harness.conflict = { detectedAt: '2026-09-03T12:00:00.000Z', kind: 'revision', serverRevision: 2 };
    const serverProject = { ...harness.project, name: 'Server project' };
    harness.resolveConflictUseServer.mockResolvedValueOnce({ project: serverProject, status: 'loaded' });
    await renderBanner();
    const useServer = Array.from(document.querySelectorAll('button')).find(
      (button) => button.textContent === 'shell.projectConflict.useServer'
    );

    await act(() => userEvent.click(useServer!));
    await act(() => userEvent.click(document.querySelector<HTMLButtonElement>('[data-testid="confirm-resolution"]')!));
    await vi.waitFor(() => expect(harness.acknowledgeProjectResolution).toHaveBeenCalledOnce());

    expect(harness.replaceProjectFromServer).toHaveBeenCalledWith({ project: serverProject, projectId: 'project-1' });
    expect(harness.replaceProjectFromServer.mock.invocationCallOrder[0]).toBeLessThan(
      harness.acknowledgeProjectResolution.mock.invocationCallOrder[0]!
    );
  });

  it('keeps the fence when use-server resolution fails', async () => {
    harness.conflict = { detectedAt: '2026-09-03T12:00:00.000Z', kind: 'revision', serverRevision: 2 };
    harness.resolveConflictUseServer.mockRejectedValueOnce(new Error('offline'));
    await renderBanner();
    const useServer = Array.from(document.querySelectorAll('button')).find(
      (button) => button.textContent === 'shell.projectConflict.useServer'
    );

    await act(() => userEvent.click(useServer!));
    await act(() => userEvent.click(document.querySelector<HTMLButtonElement>('[data-testid="confirm-resolution"]')!));
    await vi.waitFor(() => expect(harness.reportError).toHaveBeenCalledOnce());

    expect(harness.replaceProjectFromServer).not.toHaveBeenCalled();
    expect(harness.acknowledgeProjectResolution).not.toHaveBeenCalled();
  });

  it('creates a replacement before discarding the last project and then releases its fence', async () => {
    harness.conflict = { detectedAt: '2026-09-03T12:00:00.000Z', kind: 'deleted' };
    harness.closeProject.mockReturnValueOnce({ ok: false, reason: 'last-project' }).mockReturnValueOnce({ ok: true });
    await renderBanner();
    const discard = Array.from(document.querySelectorAll('button')).find(
      (button) => button.textContent === 'shell.projectConflict.discard'
    );

    await act(() => userEvent.click(discard!));
    await act(() => userEvent.click(document.querySelector<HTMLButtonElement>('[data-testid="confirm-resolution"]')!));
    await vi.waitFor(() => expect(harness.acknowledgeProjectResolution).toHaveBeenCalledOnce());

    expect(harness.createProject).toHaveBeenCalledOnce();
    expect(harness.closeProject).toHaveBeenCalledTimes(2);
    expect(harness.createProject.mock.invocationCallOrder[0]).toBeGreaterThan(
      harness.closeProject.mock.invocationCallOrder[0]!
    );
    expect(harness.createProject.mock.invocationCallOrder[0]).toBeLessThan(
      harness.closeProject.mock.invocationCallOrder[1]!
    );
    expect(harness.closeProject.mock.invocationCallOrder[1]).toBeLessThan(
      harness.acknowledgeProjectResolution.mock.invocationCallOrder[0]!
    );
  });

  it('releases the discard fence when the project cannot close', async () => {
    harness.conflict = { detectedAt: '2026-09-03T12:00:00.000Z', kind: 'deleted' };
    harness.closeProject.mockReturnValueOnce({ ok: false, reason: 'active-queue-runs' });
    await renderBanner();
    const discard = Array.from(document.querySelectorAll('button')).find(
      (button) => button.textContent === 'shell.projectConflict.discard'
    );

    await act(() => userEvent.click(discard!));
    await act(() => userEvent.click(document.querySelector<HTMLButtonElement>('[data-testid="confirm-resolution"]')!));
    await vi.waitFor(() => expect(harness.abortProjectResolution).toHaveBeenCalledWith('project-1'));

    expect(harness.acknowledgeProjectResolution).not.toHaveBeenCalled();
  });

  it('exports a newer-format draft without opening it in the editor', async () => {
    harness.recoverableDrafts = [
      { editorSessionId: 'newer-editor', generation: 1, projectId: 'future-project', updatedAt: 1 },
    ];
    await renderBanner();
    const button = Array.from(document.querySelectorAll('button')).find(
      (candidate) => candidate.textContent === 'shell.projectConflict.export'
    );

    await act(() => userEvent.click(button!));
    await vi.waitFor(() => expect(harness.downloadText).toHaveBeenCalledOnce());

    expect(harness.getRecoverableDraftDocument).toHaveBeenCalledWith('future-project', 'newer-editor');
    expect(harness.downloadText).toHaveBeenCalledWith(
      '{"documentSchemaVersion":3}',
      'future-project.json',
      'application/json'
    );
  });

  it('lets every newer-format draft be selected and exported', async () => {
    harness.recoverableDrafts = [
      { editorSessionId: 'newest-editor', generation: 2, projectId: 'newest-project', updatedAt: 2 },
      { editorSessionId: 'older-editor', generation: 1, projectId: 'older-project', updatedAt: 1 },
    ];
    await renderBanner();
    const next = Array.from(document.querySelectorAll('button')).find(
      (candidate) => candidate.textContent === 'shell.projectConflict.nextDraft'
    );
    expect(next).toBeDefined();

    await act(() => userEvent.click(next!));
    const exportButton = Array.from(document.querySelectorAll('button')).find(
      (candidate) => candidate.textContent === 'shell.projectConflict.export'
    );
    await act(() => userEvent.click(exportButton!));
    await vi.waitFor(() => expect(harness.downloadText).toHaveBeenCalledOnce());

    expect(harness.getRecoverableDraftDocument).toHaveBeenCalledWith('older-project', 'older-editor');
  });

  it('deletes a selected newer-format draft only after confirmation', async () => {
    harness.recoverableDrafts = [
      { editorSessionId: 'newer-editor', generation: 3, projectId: 'future-project', updatedAt: 17 },
    ];
    await renderBanner();
    const button = Array.from(document.querySelectorAll('button')).find(
      (candidate) => candidate.textContent === 'shell.projectConflict.deleteDraft'
    );

    await act(() => userEvent.click(button!));
    expect(harness.deleteRecoverableDraft).not.toHaveBeenCalled();
    const confirm = document.querySelector<HTMLButtonElement>('[data-testid="confirm-resolution"]');
    await act(() => userEvent.click(confirm!));

    await vi.waitFor(() =>
      expect(harness.deleteRecoverableDraft).toHaveBeenCalledWith('future-project', 'newer-editor', 3, 17)
    );
  });
});

const renderBanner = () =>
  act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <ProjectConflictBanner />
      </ChakraProvider>
    )
  );
