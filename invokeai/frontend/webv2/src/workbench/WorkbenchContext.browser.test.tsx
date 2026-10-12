import type { GenerationUiAdapter } from '@features/generation/react';
import type * as projectsApi from '@workbench/projects/api';
import type { ProjectRecordDTO } from '@workbench/projects/api';
import type { WorkbenchCommands } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { GenerationUiProvider } from '@features/generation/react';
import { useRegisterDraftFlusher } from '@platform/react/draftRegistry';
import { useMountEffect } from '@platform/react/useMountEffect';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { GenerateDenoisingStrength } from '@workbench/widgets/canvas/GenerateDenoisingStrength';
import { DEFAULT_CANVAS_DENOISING_STRENGTH } from '@workbench/widgets/canvas/invoke/canvasStrength';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/** An in-memory project server; everything between it and the editor (IndexedDB, Web Locks) is real. */
const server = vi.hoisted(() => ({
  calls: [] as string[],
  clientState: new Map<string, string>(),
  /** Writes fail as they do for requests an unloading page starts; reads still answer the next page. */
  offline: false,
  records: new Map<string, ProjectRecordDTO>(),
  updateGate: null as Promise<void> | null,
}));

vi.mock('react-i18next', () => ({ useTranslation: () => ({ i18n: { language: 'en' }, t: (key: string) => key }) }));
vi.mock('@features/intermediates/holdLease', () => ({ startIntermediatesHoldLease: () => () => undefined }));
vi.mock('@workbench/projects/api', async (importOriginal) => {
  const actual = await importOriginal<typeof projectsApi>();
  const { ApiError } = await import('@platform/transport/http');
  const notFound = () => new ApiError('not found', 404);
  const unreachable = () => Promise.reject(new TypeError('Failed to fetch'));
  return {
    ...actual,
    createProjectSettled: (request: projectsApi.ProjectCreateRequest) => {
      if (server.offline) {
        return unreachable();
      }
      const id = request.project_id!;
      const record: ProjectRecordDTO = {
        board_id: `board-${id}`,
        created_at: '2026-10-04T00:00:00.000Z',
        data: structuredClone(request.data),
        minimum_canvas_schema_version: request.minimum_canvas_schema_version ?? 3,
        name: request.name,
        project_id: id,
        revision: 1,
        updated_at: '2026-10-04T00:00:00.000Z',
      };
      server.calls.push(`create:${request.name}`);
      server.records.set(id, record);
      return Promise.resolve(structuredClone(record));
    },
    deleteClientStateValue: (key: string) => {
      server.clientState.delete(key);
      return Promise.resolve();
    },
    getClientStateValue: (key: string) => Promise.resolve(server.clientState.get(key) ?? null),
    getProject: (projectId: string) => {
      const record = server.records.get(projectId);
      return record ? Promise.resolve(structuredClone(record)) : Promise.reject(notFound());
    },
    listProjects: () => {
      server.calls.push('list');
      return Promise.resolve(
        [...server.records.values()].map(({ data: _data, ...summary }) => structuredClone(summary))
      );
    },
    setClientStateValue: (key: string, value: string) => {
      if (server.offline) {
        return unreachable();
      }
      server.clientState.set(key, value);
      return Promise.resolve();
    },
    updateProject: async (projectId: string, request: projectsApi.ProjectUpdateRequest) => {
      server.calls.push(`update:${request.name}`);
      if (server.offline) {
        return unreachable();
      }
      await server.updateGate;
      const current = server.records.get(projectId);
      if (!current) {
        throw notFound();
      }
      if (current.revision !== request.expected_revision) {
        throw new ApiError('conflict', 409);
      }
      const record: ProjectRecordDTO = {
        ...current,
        data: structuredClone(request.data),
        name: request.name,
        revision: current.revision + 1,
      };
      server.records.set(projectId, record);
      return structuredClone(record);
    },
  };
});

import { useActiveProjectSelector, useWorkbenchCommands, WorkbenchProvider } from './WorkbenchContext';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const editor = {
  commands: null as WorkbenchCommands | null,
  /** A name typed into a draft that only the draft registry commits; it has no unmount flush of its own. */
  draftName: null as string | null,
};

/** Rendered only once the editor has hydrated. */
const EditorProbe = () => {
  const commands = useWorkbenchCommands();
  const project = useActiveProjectSelector(
    (active) => ({ id: active.id, name: active.name }),
    (left, right) => left.id === right.id && left.name === right.name
  );
  useMountEffect(() => {
    editor.commands = commands;
    return () => {
      editor.commands = null;
    };
  });
  useRegisterDraftFlusher(() => {
    if (editor.draftName !== null) {
      commands.projects.rename(project.id, editor.draftName);
      editor.draftName = null;
    }
  });
  return <output data-project-id={project.id}>{project.name}</output>;
};

let host: HTMLDivElement;
let root: Root;

const activeProject = (): { id: string; name: string } | null => {
  const output = host.querySelector('output');
  return output ? { id: output.dataset.projectId!, name: output.textContent } : null;
};

const generationUi = {
  sectionPreferences: { sectionsOpen: {}, setSectionOpen: () => undefined },
} as unknown as GenerationUiAdapter;

const mountEditor = async ({ strict = false, withStrength = false } = {}) => {
  const editorTree = (
    <WorkbenchProvider>
      <EditorProbe />
      {/* A real Generate control that holds its edits in a debounced draft. */}
      {withStrength ? (
        <GenerationUiProvider adapter={generationUi}>
          <GenerateDenoisingStrength>{({ field }) => field}</GenerateDenoisingStrength>
        </GenerationUiProvider>
      ) : null}
    </WorkbenchProvider>
  );
  await act(() =>
    root.render(
      <ChakraProvider value={system}>{strict ? <StrictMode>{editorTree}</StrictMode> : editorTree}</ChakraProvider>
    )
  );
};

/** Route exit: the provider unmounts while the app (and the account) stays. */
const leaveEditor = () => act(() => root.render(<ChakraProvider value={system}>{null}</ChakraProvider>));

const waitForHydratedEditor = () => vi.waitFor(() => expect(activeProject()).not.toBeNull(), { timeout: 5_000 });

const rename = (name: string) =>
  act(() => {
    editor.commands!.projects.rename(activeProject()!.id, name);
  });

const savedName = (projectId: string) => server.records.get(projectId)?.name;

/** Every IndexedDB connection the page has opened, so an unload can cut off those that already existed. */
const openConnections = new Set<IDBDatabase>();
const openDatabase = IDBFactory.prototype.open;
const transaction = IDBDatabase.prototype.transaction;
let revivePage: (() => void) | null = null;

/**
 * What a document unload does to work the page has not finished: its IndexedDB connections start no further
 * transactions (a transaction already committed still lands) and its requests never reach the server. The next
 * page's own connections, opened afterwards, work normally.
 */
const unloadPage = (): void => {
  const deadConnections = new Set(openConnections);
  server.offline = true;
  IDBDatabase.prototype.transaction = function (this: IDBDatabase, ...args: Parameters<IDBDatabase['transaction']>) {
    if (deadConnections.has(this)) {
      throw new DOMException('The document unloaded.', 'InvalidStateError');
    }
    return transaction.apply(this, args);
  } as IDBDatabase['transaction'];
  revivePage = () => {
    IDBDatabase.prototype.transaction = transaction;
    server.offline = false;
    revivePage = null;
  };
};

beforeEach(async () => {
  IDBFactory.prototype.open = function (this: IDBFactory, ...args: Parameters<IDBFactory['open']>) {
    const request = openDatabase.apply(this, args);
    request.addEventListener('success', () => openConnections.add(request.result));
    return request;
  };
  server.calls = [];
  server.clientState.clear();
  server.records.clear();
  server.updateGate = null;
  accountLifecycle.activate('editor-exit-test', `:editor-exit-test:${crypto.randomUUID()}`);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await mountEditor();
  await waitForHydratedEditor();
  // The new project's first autosave creates it on the server.
  await vi.waitFor(() => expect(server.records.size).toBe(1), { timeout: 5_000 });
});

afterEach(async () => {
  revivePage?.();
  IDBFactory.prototype.open = openDatabase;
  openConnections.clear();
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

describe('WorkbenchProvider exit checkpoint', () => {
  it('keeps an edit made just before leaving the editor', async () => {
    const projectId = activeProject()!.id;

    await rename('Newest');
    await leaveEditor();
    await mountEditor();
    await waitForHydratedEditor();

    expect(activeProject()).toEqual({ id: projectId, name: 'Newest' });
    expect(savedName(projectId)).toBe('Newest');
  });

  it('keeps an edit made just before leaving through the development double mount', async () => {
    const projectId = activeProject()!.id;
    await leaveEditor();
    await mountEditor({ strict: true });
    await waitForHydratedEditor();

    await rename('Strict edit');
    await leaveEditor();
    await mountEditor({ strict: true });
    await waitForHydratedEditor();

    expect(activeProject()).toEqual({ id: projectId, name: 'Strict edit' });
    expect(savedName(projectId)).toBe('Strict edit');
  });

  it('commits drafts still held by editors when leaving', async () => {
    const projectId = activeProject()!.id;

    editor.draftName = 'Drafted';
    await leaveEditor();
    await mountEditor();
    await waitForHydratedEditor();

    expect(activeProject()).toEqual({ id: projectId, name: 'Drafted' });
  });

  it('saves the newest edit after a slow earlier save, before the editor loads again', async () => {
    const projectId = activeProject()!.id;
    let releaseUpdate!: () => void;
    server.updateGate = new Promise((resolve) => {
      releaseUpdate = resolve;
    });

    await rename('Slow');
    await vi.waitFor(() => expect(server.calls).toContain('update:Slow'), { timeout: 5_000 });
    await rename('Newest');
    await leaveEditor();
    await mountEditor();
    const listsBeforeRelease = server.calls.filter((call) => call === 'list').length;
    await act(async () => {
      await new Promise((resolve) => {
        globalThis.setTimeout(resolve, 200);
      });
    });

    // The remounted editor waits for the previous checkpoint instead of loading the server's older copy.
    expect(activeProject()).toBeNull();
    expect(server.calls.filter((call) => call === 'list')).toHaveLength(listsBeforeRelease);

    server.updateGate = null;
    releaseUpdate();
    await waitForHydratedEditor();

    expect(activeProject()).toEqual({ id: projectId, name: 'Newest' });
    expect(savedName(projectId)).toBe('Newest');
    const finalSave = server.calls.lastIndexOf('update:Newest');
    expect(server.calls.indexOf('update:Slow')).toBeLessThan(finalSave);
    expect(server.calls.lastIndexOf('list')).toBeGreaterThan(finalSave);
  });
});

describe('WorkbenchProvider hidden page', () => {
  const strengthSlider = () => host.querySelector<HTMLElement>('[data-scope="scrubber"] [role="slider"]');
  const strengthValue = () => Number(strengthSlider()?.getAttribute('aria-valuenow'));
  const typedStrength = DEFAULT_CANVAS_DENOISING_STRENGTH + 0.01;

  it('saves an edit still held in a draft when the page is hidden, before the draft debounce', async () => {
    const projectId = activeProject()!.id;
    await leaveEditor();
    await mountEditor({ withStrength: true });
    await vi.waitFor(() => expect(strengthSlider()).not.toBeNull(), { timeout: 5_000 });
    const savedBeforeEdit = JSON.stringify(server.records.get(projectId)?.data);

    const slider = strengthSlider()!;
    await act(() => {
      slider.focus();
      slider.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
    });
    act(() => {
      window.dispatchEvent(new PageTransitionEvent('pagehide'));
    });

    // Committing the draft leaves the control as the user left it.
    expect(document.activeElement).toBe(slider);
    expect(strengthValue()).toBeCloseTo(typedStrength);
    // Well before the 250 ms draft debounce and 500 ms autosave could have saved it on their own.
    await vi.waitFor(() => expect(JSON.stringify(server.records.get(projectId)?.data)).not.toBe(savedBeforeEdit), {
      timeout: 600,
    });

    // The page goes away: its lifetime ends without the editor's exit checkpoint, and a new one loads it again.
    const { storageSuffix } = accountLifecycle.capture();
    accountLifecycle.invalidate();
    await leaveEditor();
    accountLifecycle.activate('editor-exit-test', storageSuffix);
    await mountEditor({ withStrength: true });
    await vi.waitFor(() => expect(strengthSlider()).not.toBeNull(), { timeout: 5_000 });

    expect(activeProject()!.id).toBe(projectId);
    expect(strengthValue()).toBeCloseTo(typedStrength);
  });

  it('recovers an edit still held in a draft when the page unloads before anything can save it', async () => {
    const projectId = activeProject()!.id;
    await leaveEditor();
    await mountEditor({ withStrength: true });
    await vi.waitFor(() => expect(strengthSlider()).not.toBeNull(), { timeout: 5_000 });
    const savedBeforeEdit = JSON.stringify(server.records.get(projectId)?.data);

    const slider = strengthSlider()!;
    await act(() => {
      slider.focus();
      slider.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
    });
    act(() => {
      window.dispatchEvent(new PageTransitionEvent('pagehide'));
      unloadPage();
    });
    // The page's state goes away; neither its hidden-page save nor its exit checkpoint can write anything.
    await leaveEditor();
    await mountEditor({ withStrength: true });
    await vi.waitFor(() => expect(strengthSlider()).not.toBeNull(), { timeout: 5_000 });

    expect(activeProject()!.id).toBe(projectId);
    expect(strengthValue()).toBeCloseTo(typedStrength);
    expect(JSON.stringify(server.records.get(projectId)?.data)).toBe(savedBeforeEdit);
    // Recovered as a pending edit, which the next page saves once it can.
    revivePage!();
    await vi.waitFor(() => expect(JSON.stringify(server.records.get(projectId)?.data)).not.toBe(savedBeforeEdit), {
      timeout: 5_000,
    });
  });

  it('does not let the journal of a back/forward-cached page override work saved after it resumed', async () => {
    const projectId = activeProject()!.id;

    await rename('While cached');
    act(() => {
      window.dispatchEvent(new PageTransitionEvent('pagehide', { persisted: true }));
    });
    // Restored from the cache: the save requested on hiding lands, and editing continues.
    act(() => {
      window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }));
    });
    await vi.waitFor(() => expect(savedName(projectId)).toBe('While cached'), { timeout: 5_000 });
    await rename('After resuming');
    await vi.waitFor(() => expect(savedName(projectId)).toBe('After resuming'), { timeout: 5_000 });

    act(unloadPage);
    await leaveEditor();
    revivePage!();
    await mountEditor();
    await waitForHydratedEditor();

    expect(activeProject()).toEqual({ id: projectId, name: 'After resuming' });
  });
});
