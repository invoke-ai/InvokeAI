import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowRecordDTO } from '@features/workflow/data/api';
import type { WorkflowReadPort, WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowLibraryWriteRefusedError } from '@features/workflow/data/api';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { requestWorkflowPublication, workflowUiStore } from '@features/workflow/ui/workflowUiStore';
import { createProjectGraph, serializeWorkflowJson } from '@features/workflow/utility';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkflowPublicationHost } from './WorkflowPublicationHost';

/**
 * The publication controller reaches the transport through the API module itself (not the queries barrel), so
 * that is where the library is stubbed; the barrel re-exports the same stubs to the host's review fetch.
 */
const api = vi.hoisted(() => ({
  createLibraryWorkflowRecord: vi.fn(),
  getLibraryWorkflowRecord: vi.fn(),
  updateLibraryWorkflow: vi.fn(),
}));

vi.mock('@features/workflow/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ...api,
}));

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useInvocationTemplatesSnapshot: () => ({ error: null, status: 'loaded', templates: {} }),
}));

const TRANSLATIONS: Record<string, string> = {
  'common.cancel': 'Cancel',
  'workflowLibrary.conflictStaleBody':
    'The library template behind "{{name}}" has changed since this copy was loaded or last saved.',
  'workflowLibrary.conflictTitle': 'Template has changed',
  'workflowLibrary.conflictUnknownBody':
    'It is not known whether the library template behind "{{name}}" still matches this copy.',
  'workflowLibrary.replaceTemplate': 'Replace template',
  'workflowLibrary.retry': 'Retry',
  'workflowLibrary.retryBody':
    'Saving "{{name}}" did not get an answer from the library ({{message}}). Retry the same save?',
  'workflowLibrary.retryDiscard': 'Discard',
  'workflowLibrary.retryTitle': 'Library did not answer',
  'workflowLibrary.reviewBody':
    'This is the current library template. Replacing it writes this project workflow over it.',
  'workflowLibrary.reviewFailed': 'The current template could not be loaded.',
  'workflowLibrary.reviewLoading': 'Loading the current template…',
  'workflowLibrary.reviewRevision': 'Revision {{revision}} · updated {{when}}',
  'workflowLibrary.reviewTemplate': 'Review current template',
  'workflowLibrary.reviewTitle': 'Current library template',
  'workflowLibrary.saveAsNew': 'Save as new',
  'workflowLibrary.saveFailed': 'Failed to save workflow',
  'workflowLibrary.saveToLibraryConfirm': 'Save',
  'workflowLibrary.saveToLibraryExplanation':
    "Creates a new library template from this workflow. Its current input values become the template's defaults.",
  'workflowLibrary.saveToLibraryTitle': 'Save to library',
  'workflowLibrary.saved': 'Workflow saved',
  'workflowLibrary.savedCreatedBody': 'Saved "{{name}}" to the library.',
  'workflowLibrary.savedUpdatedBody': 'Updated "{{name}}" in the library.',
  'workflowLibrary.templateName': 'Template name',
  'workflowLibrary.unavailableBundledBody':
    '"{{name}}" came from a bundled template, which cannot be changed. Save it as a new template instead.',
  'workflowLibrary.unavailableTitle': 'Template cannot be updated',
  'workflowLibrary.untitled': 'Untitled Workflow',
  'workflowLibrary.updateConfirm': 'Update template',
  'workflowLibrary.updateConfirmBody':
    "Replace this workflow's library template with its current graph, form, and input values?",
  'workflowLibrary.updateConfirmCallers': 'Workflows that call this template will use the new version.',
  'workflowLibrary.updateConfirmNamedBody':
    'Replace the library template "{{template}}" with this workflow\'s current graph, form, and input values?',
  'workflowLibrary.updateTitle': 'Update library template',
};

const interpolate = (template: string, options?: Record<string, unknown>): string =>
  options ? template.replaceAll(/\{\{(\w+)\}\}/g, (_match, key: string) => String(options[key] ?? '')) : template;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) => interpolate(TRANSLATIONS[key] ?? key, options),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createMutablePort = <Snapshot,>(initialSnapshot: Snapshot) => {
  let snapshot = initialSnapshot;
  const listeners = new Set<() => void>();
  const port: WorkflowReadPort<Snapshot> = {
    getSnapshot: () => snapshot,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  return {
    port,
    setSnapshot: (next: Snapshot) => {
      snapshot = next;
      for (const listener of listeners) {
        listener();
      }
    },
  };
};

const PROJECT_ID = 'project-1';
const WORKFLOW_ID = 'wf-1';
const LIBRARY_ID = 'lib-1';

const UNLINKED: ProjectWorkflowEntry = { document: { ...createProjectGraph(WORKFLOW_ID), name: 'Alpha' } };
const LINKED: ProjectWorkflowEntry = { ...UNLINKED, source: { libraryWorkflowId: LIBRARY_ID, revision: 3 } };
const LINKED_UNKNOWN: ProjectWorkflowEntry = { ...UNLINKED, source: { libraryWorkflowId: LIBRARY_ID, revision: null } };
const BUNDLED: ProjectWorkflowEntry = { ...UNLINKED, source: { libraryWorkflowId: 'default_alpha', revision: 1 } };
const OTHER: ProjectWorkflowEntry = { document: { ...createProjectGraph('wf-2'), name: 'Beta' } };

const projectSnapshot = (workflows: readonly ProjectWorkflowEntry[]) => ({
  activeWorkflow: workflows[0]!,
  activeWorkflowId: workflows[0]!.document.id,
  galleryValues: {},
  id: PROJECT_ID,
  isWorkflowRunning: false,
  projectGraph: workflows[0]!.document,
  workflowValues: {},
  workflows,
});

const record = (overrides: Partial<WorkflowRecordDTO> = {}): WorkflowRecordDTO => ({
  category: 'user',
  description: '',
  name: 'Alpha',
  revision: 4,
  updated_at: new Date().toISOString(),
  workflow: serializeWorkflowJson({ ...createProjectGraph('lib-doc'), name: 'Alpha' }),
  workflow_id: LIBRARY_ID,
  ...overrides,
});

const deferred = <T,>() => {
  let reject!: (reason?: unknown) => void;
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });

  return { promise, reject, resolve };
};

describe('WorkflowPublicationHost', () => {
  let host: HTMLDivElement;
  let root: Root;
  let project: ReturnType<typeof createMutablePort<ReturnType<typeof projectSnapshot>>>;
  let commands: { renameWorkflow: ReturnType<typeof vi.fn>; setWorkflowSource: ReturnType<typeof vi.fn> };
  let notifications: {
    error: ReturnType<typeof vi.fn>;
    info: ReturnType<typeof vi.fn>;
    success: ReturnType<typeof vi.fn>;
  };

  const settleFrame = () =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

  /** Lets the async publication chain and the dialogs' presence transitions settle inside act. */
  const flush = () =>
    act(async () => {
      await settleFrame();
      await settleFrame();
    });

  const renderHost = async (workflows: readonly ProjectWorkflowEntry[]) => {
    project = createMutablePort(projectSnapshot(workflows));
    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally stable for this render lifetime
    const adapter = {
      commands,
      getProjectGraph: () => project.port.getSnapshot().projectGraph,
      notifications,
      project: project.port,
    } as unknown as WorkflowUiAdapter;

    await act(async () => {
      root.render(
        <StrictMode>
          <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
            <ChakraProvider value={system}>
              <WorkflowUiProvider adapter={adapter}>
                <WorkflowPublicationHost />
              </WorkflowUiProvider>
            </ChakraProvider>
          </QueryClientProvider>
        </StrictMode>
      );
      await settleFrame();
    });
  };

  const request = async (kind: 'save-as-new' | 'update-source', workflowId = WORKFLOW_ID) => {
    await act(async () => {
      requestWorkflowPublication({ kind, workflowId });
      await settleFrame();
    });
  };

  // A closed dialog stays mounted through its exit transition; only an open one counts as shown.
  const dialogs = () => [
    ...document.querySelectorAll<HTMLElement>(
      '[role="dialog"][data-state="open"], [role="alertdialog"][data-state="open"]'
    ),
  ];
  const dialogTitled = (title: string) =>
    dialogs().find((dialog) => dialog.querySelector('h2')?.textContent === title) ?? null;
  const saveDialog = () => document.querySelector<HTMLElement>('[data-save-to-library-dialog][data-state="open"]');
  const reviewDialog = () => document.querySelector<HTMLElement>('[data-workflow-review-dialog][data-state="open"]');
  const choice = (value: string) => document.querySelector<HTMLButtonElement>(`[data-choice="${value}"]`);
  const buttonWithText = (scope: ParentNode | null, text: string) =>
    [...(scope?.querySelectorAll('button') ?? [])].find((candidate) => (candidate.textContent ?? '').trim() === text);

  const click = async (element: HTMLElement | null | undefined) => {
    expect(element, 'expected an element to click').not.toBeFalsy();

    await act(async () => {
      element?.click();
      await settleFrame();
    });
    await flush();
  };

  const submitSaveAsNew = async (name: string) => {
    const dialog = saveDialog();
    expect(dialog).not.toBeNull();

    const input = dialog?.querySelector<HTMLInputElement>('input[name="templateName"]');
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

    await act(async () => {
      setValue?.call(input, name);
      input?.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });
    await click(buttonWithText(dialog, 'Save'));
  };

  const confirmUpdate = async () => {
    const dialog = dialogTitled('Update library template');
    expect(dialog).not.toBeNull();
    await click(buttonWithText(dialog, 'Update template'));
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    commands = { renameWorkflow: vi.fn(), setWorkflowSource: vi.fn() };
    notifications = { error: vi.fn(), info: vi.fn(), success: vi.fn() };
    api.createLibraryWorkflowRecord.mockReset();
    api.getLibraryWorkflowRecord.mockReset();
    api.updateLibraryWorkflow.mockReset();
    // A fresh controller and an empty intent for every test.
    accountLifecycle.invalidate();
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('saves the workflow as a new template under the typed name and links the copy to it', async () => {
    api.createLibraryWorkflowRecord.mockImplementation((workflow: Record<string, unknown>) =>
      Promise.resolve(record({ name: String(workflow.name), revision: 1, workflow, workflow_id: 'lib-new' }))
    );

    await renderHost([UNLINKED, OTHER]);
    await request('save-as-new');

    expect(saveDialog()?.querySelector<HTMLInputElement>('input[name="templateName"]')?.value).toBe('Alpha');
    expect(workflowUiStore.getSnapshot().publicationIntent).toBeNull();

    await submitSaveAsNew('Alpha, published');

    expect(api.createLibraryWorkflowRecord).toHaveBeenCalledTimes(1);
    const [workflow, options] = api.createLibraryWorkflowRecord.mock.calls[0] as [
      Record<string, unknown>,
      { reservedId?: string },
    ];
    expect(workflow).toMatchObject({ ...serializeWorkflowJson(UNLINKED.document), name: 'Alpha, published' });
    expect(options.reservedId).toEqual(expect.any(String));
    // The project workflow now knows which template, at which revision, it can update later.
    expect(commands.setWorkflowSource).toHaveBeenCalledWith(
      { projectId: PROJECT_ID, workflowId: WORKFLOW_ID },
      { libraryWorkflowId: 'lib-new', revision: 1 }
    );
    // It is now that template's working copy, so it takes the template's name.
    expect(commands.renameWorkflow).toHaveBeenCalledWith(WORKFLOW_ID, 'Alpha, published', PROJECT_ID);
    expect(notifications.success).toHaveBeenCalledWith('Workflow saved', 'Saved "Alpha, published" to the library.');
    expect(saveDialog()).toBeNull();
  });

  it('updates the template a save as new linked, naming it in the confirmation', async () => {
    api.createLibraryWorkflowRecord.mockImplementation((workflow: Record<string, unknown>) =>
      Promise.resolve(record({ name: String(workflow.name), revision: 1, workflow, workflow_id: 'lib-demo' }))
    );
    api.getLibraryWorkflowRecord.mockImplementation((id: string) =>
      Promise.resolve(
        record({ name: id === 'lib-demo' ? 'demo' : 'Library alpha', revision: id === 'lib-demo' ? 1 : 3 })
      )
    );
    api.updateLibraryWorkflow.mockResolvedValue(record({ name: 'demo', revision: 2, workflow_id: 'lib-demo' }));
    // Opened from a library template, then saved as a new one: the app re-links the copy to what was saved.
    commands.setWorkflowSource.mockImplementation((_target, source: ProjectWorkflowEntry['source']) =>
      project.setSnapshot(projectSnapshot([{ ...LINKED, source }]))
    );

    await renderHost([LINKED]);
    await request('save-as-new');
    await submitSaveAsNew('demo');
    await request('update-source');
    await flush();

    const dialog = dialogTitled('Update library template');
    // Only the template being replaced is named, never the one the workflow was first opened from.
    expect(dialog?.textContent).toContain(
      'Replace the library template "demo" with this workflow\'s current graph, form, and input values?'
    );
    expect(dialog?.textContent).not.toContain('Alpha');
    await confirmUpdate();

    expect(api.updateLibraryWorkflow).toHaveBeenCalledTimes(1);
    expect(api.updateLibraryWorkflow.mock.calls[0]![0]).toBe('lib-demo');
    expect(api.updateLibraryWorkflow.mock.calls[0]![2]).toMatchObject({ expectedRevision: 1 });
  });

  it('updates the template at the revision the copy knows, after confirmation, keeping its own name', async () => {
    api.getLibraryWorkflowRecord.mockResolvedValue(record({ name: 'Library alpha', revision: 3 }));
    api.updateLibraryWorkflow.mockResolvedValue(record({ name: 'Library alpha', revision: 4 }));

    await renderHost([LINKED]);
    await request('update-source');

    const dialog = dialogTitled('Update library template');
    await flush();
    expect(dialog?.textContent).toContain('Replace the library template "Library alpha" with');
    expect(api.updateLibraryWorkflow).not.toHaveBeenCalled();

    await confirmUpdate();

    expect(api.updateLibraryWorkflow).toHaveBeenCalledTimes(1);
    const [libraryId, workflow, options] = api.updateLibraryWorkflow.mock.calls[0] as [
      string,
      Record<string, unknown>,
      { expectedRevision?: number },
    ];
    expect(libraryId).toBe(LIBRARY_ID);
    expect(workflow).toEqual({ ...serializeWorkflowJson(LINKED.document), name: 'Library alpha' });
    expect(options.expectedRevision).toBe(3);
    expect(commands.setWorkflowSource).toHaveBeenCalledWith(
      { projectId: PROJECT_ID, workflowId: WORKFLOW_ID },
      { libraryWorkflowId: LIBRARY_ID, revision: 4 }
    );
    expect(notifications.success).toHaveBeenCalledWith('Workflow saved', 'Updated "Library alpha" in the library.');
    expect(dialogTitled('Update library template')).toBeNull();
  });

  it('surfaces a revision conflict after a confirmed update instead of closing silently', async () => {
    api.getLibraryWorkflowRecord.mockResolvedValue(record({ revision: 3 }));
    api.updateLibraryWorkflow.mockRejectedValue(new WorkflowLibraryWriteRefusedError('revision-conflict', 'stale', 5));

    await renderHost([LINKED]);
    await request('update-source');
    await confirmUpdate();

    expect(api.updateLibraryWorkflow).toHaveBeenCalledTimes(1);
    expect(commands.setWorkflowSource).not.toHaveBeenCalled();
    expect(notifications.success).not.toHaveBeenCalled();

    const conflict = dialogTitled('Template has changed');
    expect(conflict?.textContent).toContain('has changed since this copy was loaded or last saved');
    expect(choice('save-as-new')).not.toBeNull();
    expect(choice('review')).not.toBeNull();
  });

  it('asks before writing when the copy never knew its template revision, and reviews before replacing', async () => {
    api.getLibraryWorkflowRecord.mockResolvedValue(record({ description: 'Edited elsewhere', revision: 5 }));
    api.updateLibraryWorkflow.mockResolvedValue(record({ revision: 6 }));

    await renderHost([LINKED_UNKNOWN]);
    await request('update-source');

    const conflict = dialogTitled('Template has changed');
    expect(conflict?.textContent).toContain('It is not known whether the library template behind "Alpha"');
    expect(api.updateLibraryWorkflow).not.toHaveBeenCalled();

    await click(choice('review'));

    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledWith(LIBRARY_ID, expect.anything());
    expect(reviewDialog()?.textContent).toContain('Edited elsewhere');
    expect(reviewDialog()?.textContent).toContain('Revision 5');
    expect(api.updateLibraryWorkflow).not.toHaveBeenCalled();

    await click(buttonWithText(reviewDialog(), 'Replace template'));
    await confirmUpdate();

    // The write is fenced at exactly the revision that was reviewed.
    expect(api.updateLibraryWorkflow).toHaveBeenCalledTimes(1);
    expect(api.updateLibraryWorkflow.mock.calls[0]?.[2]).toMatchObject({ expectedRevision: 5 });
    expect(commands.setWorkflowSource).toHaveBeenCalledWith(
      { projectId: PROJECT_ID, workflowId: WORKFLOW_ID },
      { libraryWorkflowId: LIBRARY_ID, revision: 6 }
    );
    expect(notifications.success).toHaveBeenCalledWith('Workflow saved', 'Updated "Alpha" in the library.');
    expect(dialogs()).toHaveLength(0);
  });

  it('lets a conflict be resolved by saving as a new template instead', async () => {
    await renderHost([LINKED_UNKNOWN]);
    await request('update-source');

    await click(choice('save-as-new'));

    expect(dialogTitled('Template has changed')).toBeNull();
    expect(saveDialog()).not.toBeNull();
    expect(api.updateLibraryWorkflow).not.toHaveBeenCalled();
  });

  it('refuses to update a bundled template and offers saving as new instead', async () => {
    await renderHost([BUNDLED]);
    await request('update-source');

    expect(dialogTitled('Template cannot be updated')?.textContent).toContain('came from a bundled template');
    expect(api.updateLibraryWorkflow).not.toHaveBeenCalled();
  });

  it('offers a retry after a lost answer and resends under the same reserved id', async () => {
    api.createLibraryWorkflowRecord
      .mockRejectedValueOnce(new Error('offline'))
      .mockImplementationOnce((workflow: Record<string, unknown>) =>
        Promise.resolve(record({ name: String(workflow.name), revision: 1, workflow, workflow_id: 'lib-new' }))
      );

    await renderHost([UNLINKED]);
    await request('save-as-new');
    await submitSaveAsNew('Alpha');

    expect(dialogTitled('Library did not answer')?.textContent).toContain('(offline)');
    expect(notifications.success).not.toHaveBeenCalled();

    await click(choice('retry'));

    expect(api.createLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
    const first = api.createLibraryWorkflowRecord.mock.calls[0]?.[1] as { reservedId: string };
    const second = api.createLibraryWorkflowRecord.mock.calls[1]?.[1] as { reservedId: string };
    // The server matches the resend against whatever the first send created, so nothing is saved twice.
    expect(second.reservedId).toBe(first.reservedId);
    expect(dialogs()).toHaveLength(0);
  });

  it('links the copy and reports success once a retried save lands', async () => {
    api.createLibraryWorkflowRecord
      .mockRejectedValueOnce(new Error('offline'))
      .mockImplementationOnce((workflow: Record<string, unknown>) =>
        Promise.resolve(record({ name: String(workflow.name), revision: 1, workflow, workflow_id: 'lib-new' }))
      );

    await renderHost([UNLINKED]);
    await request('save-as-new');
    await submitSaveAsNew('Alpha');
    await click(choice('retry'));

    expect(commands.setWorkflowSource).toHaveBeenCalledWith(
      { projectId: PROJECT_ID, workflowId: WORKFLOW_ID },
      { libraryWorkflowId: 'lib-new', revision: 1 }
    );
    expect(notifications.success).toHaveBeenCalledWith('Workflow saved', 'Saved "Alpha" to the library.');
  });

  it("never lets an earlier project's late answer steer a publication opened in another project", async () => {
    const pending = deferred<WorkflowRecordDTO>();
    api.getLibraryWorkflowRecord.mockResolvedValue(record({ revision: 3 }));
    api.updateLibraryWorkflow.mockReturnValueOnce(pending.promise);

    await renderHost([LINKED]);
    await request('update-source');
    await confirmUpdate();
    expect(api.updateLibraryWorkflow).toHaveBeenCalledTimes(1);

    // A duplicate of the project shares its workflow ids but points at a different template.
    const copied: ProjectWorkflowEntry = {
      ...LINKED,
      document: { ...LINKED.document, description: 'Only project B content', name: 'Beta project copy' },
      source: { libraryWorkflowId: 'lib-B', revision: 1 },
    };
    await act(() => project.setSnapshot({ ...projectSnapshot([copied]), id: 'project-2' }));
    await request('save-as-new');
    expect(saveDialog()).not.toBeNull();

    await act(async () => {
      pending.reject(new WorkflowLibraryWriteRefusedError('revision-conflict', 'stale', 5));
      await settleFrame();
    });
    await flush();

    // The late conflict belongs to the ended operation: the save dialog stays, no conflict dialog appears.
    expect(saveDialog()).not.toBeNull();
    expect(dialogTitled('Template has changed')).toBeNull();
    expect(saveDialog()?.querySelector<HTMLButtonElement>('button[type="submit"]')?.disabled).toBe(false);

    api.createLibraryWorkflowRecord.mockResolvedValue(record({ name: 'Beta', workflow_id: 'lib-C' }));
    await submitSaveAsNew('Beta');
    expect(api.createLibraryWorkflowRecord).toHaveBeenCalledTimes(1);
    expect(api.updateLibraryWorkflow).toHaveBeenCalledTimes(1);
    expect(commands.setWorkflowSource).toHaveBeenCalledWith(
      { projectId: 'project-2', workflowId: WORKFLOW_ID },
      { libraryWorkflowId: 'lib-C', revision: expect.any(Number) }
    );
  });

  it("keeps another project's open confirmation when an earlier update settles", async () => {
    const pending = deferred<WorkflowRecordDTO>();
    api.getLibraryWorkflowRecord.mockResolvedValue(record({ revision: 3 }));
    api.updateLibraryWorkflow.mockReturnValueOnce(pending.promise);

    await renderHost([LINKED]);
    await request('update-source');
    await confirmUpdate();

    // Project B is a duplicate: same workflow id, its own template at its own revision.
    const copied: ProjectWorkflowEntry = { ...LINKED, source: { libraryWorkflowId: 'lib-B', revision: 1 } };
    await act(() => project.setSnapshot({ ...projectSnapshot([copied]), id: 'project-2' }));
    await request('update-source');
    expect(dialogTitled('Update library template')).not.toBeNull();

    await act(async () => {
      pending.resolve(record({ revision: 4 }));
      await settleFrame();
    });
    await flush();

    // A's settled confirmation closed A's dialog only; B's confirmation is still waiting for its answer.
    expect(dialogTitled('Update library template')).not.toBeNull();
    api.getLibraryWorkflowRecord.mockResolvedValue(record({ revision: 1, workflow_id: 'lib-B' }));
    api.updateLibraryWorkflow.mockResolvedValueOnce(record({ revision: 2, workflow_id: 'lib-B' }));
    await confirmUpdate();
    expect(api.updateLibraryWorkflow).toHaveBeenLastCalledWith(
      'lib-B',
      expect.anything(),
      expect.objectContaining({ expectedRevision: 1 })
    );
  });

  it("keeps another project's confirmation usable while an earlier update is still pending", async () => {
    const pending = deferred<WorkflowRecordDTO>();
    api.getLibraryWorkflowRecord.mockResolvedValue(record({ revision: 3 }));
    api.updateLibraryWorkflow.mockReturnValueOnce(pending.promise);

    await renderHost([LINKED]);
    await request('update-source');
    await confirmUpdate();

    const copied: ProjectWorkflowEntry = { ...LINKED, source: { libraryWorkflowId: 'lib-B', revision: 1 } };
    await act(() => project.setSnapshot({ ...projectSnapshot([copied]), id: 'project-2' }));
    await request('update-source');

    // B's confirmation is its own dialog: nothing of A's in-flight state reaches it.
    const dialog = dialogTitled('Update library template');
    expect(buttonWithText(dialog, 'Update template')?.disabled).toBe(false);
    expect(buttonWithText(dialog, 'Cancel')?.disabled).toBe(false);

    api.getLibraryWorkflowRecord.mockResolvedValue(record({ revision: 1, workflow_id: 'lib-B' }));
    api.updateLibraryWorkflow.mockResolvedValueOnce(record({ revision: 2, workflow_id: 'lib-B' }));
    await confirmUpdate();
    expect(api.updateLibraryWorkflow).toHaveBeenLastCalledWith('lib-B', expect.anything(), expect.anything());
    expect(dialogTitled('Update library template')).toBeNull();

    await act(async () => {
      pending.resolve(record({ revision: 4 }));
      await settleFrame();
    });
    await flush();
    expect(dialogs()).toHaveLength(0);
  });

  it('offers a lost save again after switching away and back, and resends it under the same reserved id', async () => {
    api.createLibraryWorkflowRecord
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockImplementationOnce((workflow: Record<string, unknown>, options: { reservedId: string }) =>
        // The first send did land: the server answers the resend with the record it already holds.
        Promise.resolve(record({ name: String(workflow.name), revision: 1, workflow, workflow_id: options.reservedId }))
      );

    await renderHost([UNLINKED]);
    await request('save-as-new');
    await submitSaveAsNew('Alpha');
    expect(dialogTitled('Library did not answer')).not.toBeNull();

    await act(() => project.setSnapshot({ ...projectSnapshot([OTHER]), id: 'project-2' }));
    await flush();
    expect(dialogs()).toHaveLength(0);

    await act(() => project.setSnapshot(projectSnapshot([UNLINKED])));
    await flush();
    expect(dialogTitled('Library did not answer')).not.toBeNull();

    await click(choice('retry'));

    expect(api.createLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
    const first = api.createLibraryWorkflowRecord.mock.calls[0]?.[1] as { reservedId: string };
    const second = api.createLibraryWorkflowRecord.mock.calls[1]?.[1] as { reservedId: string };
    expect(second.reservedId).toBe(first.reservedId);
    expect(commands.setWorkflowSource).toHaveBeenCalledWith(
      { projectId: PROJECT_ID, workflowId: WORKFLOW_ID },
      { libraryWorkflowId: first.reservedId, revision: 1 }
    );
    expect(dialogs()).toHaveLength(0);

    // Settled: coming back again offers nothing.
    await act(() => project.setSnapshot({ ...projectSnapshot([OTHER]), id: 'project-2' }));
    await act(() => project.setSnapshot(projectSnapshot([UNLINKED])));
    await flush();
    expect(dialogs()).toHaveLength(0);
  });

  it('offers a lost save once per return, and lets the user discard it to save afresh', async () => {
    api.createLibraryWorkflowRecord
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockImplementation((workflow: Record<string, unknown>, options: { reservedId: string }) =>
        Promise.resolve(record({ name: String(workflow.name), revision: 1, workflow, workflow_id: options.reservedId }))
      );

    await renderHost([UNLINKED]);
    await request('save-as-new');
    await submitSaveAsNew('Alpha');
    expect(dialogTitled('Library did not answer')).not.toBeNull();

    // Cancel defers: the handle is kept and offered once more on the next return, then stays quiet.
    await click(buttonWithText(dialogTitled('Library did not answer'), 'Cancel'));
    expect(dialogs()).toHaveLength(0);
    await act(() => project.setSnapshot({ ...projectSnapshot([OTHER]), id: 'project-2' }));
    await act(() => project.setSnapshot(projectSnapshot([UNLINKED])));
    await flush();
    expect(dialogTitled('Library did not answer')).not.toBeNull();
    await click(buttonWithText(dialogTitled('Library did not answer'), 'Cancel'));
    await act(() => project.setSnapshot({ ...projectSnapshot([OTHER]), id: 'project-2' }));
    await act(() => project.setSnapshot(projectSnapshot([UNLINKED])));
    await flush();
    expect(dialogs()).toHaveLength(0);

    // A new save for that workflow meets the unanswered one first; discarding it lets the new save proceed.
    await request('save-as-new');
    expect(dialogTitled('Library did not answer')).not.toBeNull();
    expect(saveDialog()).toBeNull();
    await click(choice('discard'));
    expect(dialogs()).toHaveLength(0);

    await request('save-as-new');
    expect(saveDialog()).not.toBeNull();
    await submitSaveAsNew('Beta');
    expect(api.createLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
    const first = api.createLibraryWorkflowRecord.mock.calls[0]?.[1] as { reservedId: string };
    const second = api.createLibraryWorkflowRecord.mock.calls[1]?.[1] as { reservedId: string };
    expect(second.reservedId).not.toBe(first.reservedId);
    expect(notifications.success).toHaveBeenCalledWith('Workflow saved', 'Saved "Beta" to the library.');
  });

  it('ends the stage when its workflow leaves the project', async () => {
    const pending = deferred<WorkflowRecordDTO>();
    api.createLibraryWorkflowRecord.mockReturnValue(pending.promise);

    await renderHost([UNLINKED, OTHER]);
    await request('save-as-new');
    expect(saveDialog()).not.toBeNull();

    await act(() => project.setSnapshot(projectSnapshot([OTHER])));
    await flush();

    expect(saveDialog()).toBeNull();

    // A request for a workflow the project does not hold starts nothing.
    await request('save-as-new', WORKFLOW_ID);
    expect(saveDialog()).toBeNull();
    expect(workflowUiStore.getSnapshot().publicationIntent).toBeNull();
  });
});
