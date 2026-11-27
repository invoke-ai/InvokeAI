import type { ProjectWorkflowEntry } from '@features/workflow/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import {
  requestLibraryCopyChoice,
  setWorkflowLibraryOpen,
  workflowUiStore,
} from '@features/workflow/ui/workflowUiStore';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { LibraryCopyChoiceHost } from './LibraryCopyChoiceDialog';

const loader = vi.hoisted(() => ({
  open: vi.fn((_item: unknown, _mode: string) => Promise.resolve()),
  replace: vi.fn((_item: unknown, _workflowId: string) => Promise.resolve()),
  resume: vi.fn((_workflowId: string) => {}),
}));

vi.mock('./useOpenLibraryWorkflow', () => ({
  useOpenLibraryWorkflow: (onOpened: () => void) => ({
    loadPhase: 'idle',
    open: loader.open,
    replace: loader.replace,
    resume: (workflowId: string) => {
      loader.resume(workflowId);
      onOpened();
    },
  }),
}));

const project = vi.hoisted(() => ({
  activeWorkflowId: 'copy-1',
  id: 'project-1',
  listeners: new Set<() => void>(),
  workflows: [] as ProjectWorkflowEntry[],
}));

vi.mock('@features/workflow/ui/WorkflowUiContext', () => ({
  useWorkflowProjectSelector: (selector: (snapshot: typeof project) => unknown) => selector(project),
  useWorkflowUi: () => ({
    project: {
      getSnapshot: () => project,
      subscribe: (listener: () => void) => {
        project.listeners.add(listener);
        return () => project.listeners.delete(listener);
      },
    },
  }),
}));

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ITEM = { name: 'Portrait', workflow_id: 'lib-portrait' };
const copy = (id: string, name: string, libraryWorkflowId = ITEM.workflow_id): ProjectWorkflowEntry => ({
  document: { ...createProjectGraph(id), name },
  source: { libraryWorkflowId, revision: 1 },
});

const dialog = () => document.querySelector<HTMLElement>('[data-library-copy-choice]');
const button = (label: string) =>
  [...(dialog()?.querySelectorAll<HTMLButtonElement>('button') ?? [])].find(
    (candidate) => candidate.textContent === label
  );

describe('LibraryCopyChoiceHost', () => {
  let host: HTMLDivElement;
  let root: Root;

  const render = () =>
    act(() =>
      root.render(
        <ChakraProvider value={system}>
          <LibraryCopyChoiceHost />
        </ChakraProvider>
      )
    );

  beforeEach(() => {
    loader.open.mockClear();
    loader.replace.mockClear();
    loader.resume.mockClear();
    project.id = 'project-1';
    project.activeWorkflowId = 'copy-1';
    project.workflows = [copy('copy-1', 'Portrait'), copy('other', 'Other', 'lib-other')];
    workflowUiStore.patchSnapshot({ libraryCopyChoice: null });
    setWorkflowLibraryOpen(true);
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('asks about the one copy; opening it switches there and closes the library', async () => {
    await render();
    expect(dialog()).toBeNull();

    await act(() => requestLibraryCopyChoice('project-1', ITEM));

    await vi.waitFor(() => expect(dialog()).not.toBeNull());
    expect(dialog()?.querySelector('[role="combobox"]')).toBeNull();
    // Enter on arrival opens the copy; it must never replace it.
    await vi.waitFor(() => expect(document.activeElement).toBe(button('workflowLibrary.copyChoice.open')));

    await act(() => button('workflowLibrary.copyChoice.open')!.click());

    expect(loader.resume).toHaveBeenCalledExactlyOnceWith('copy-1');
    expect(workflowUiStore.getSnapshot().libraryCopyChoice).toBeNull();
    expect(workflowUiStore.getSnapshot().isLibraryOpen).toBe(false);
  });

  it('adds another copy, or replaces the copy picked from several', async () => {
    project.workflows = [copy('copy-1', 'Portrait'), copy('copy-2', 'Portrait, tuned')];
    await render();
    await act(() => requestLibraryCopyChoice('project-1', ITEM));
    await vi.waitFor(() => expect(dialog()).not.toBeNull());

    await act(() => button('workflowLibrary.copyChoice.addCopy')!.click());
    expect(loader.open).toHaveBeenCalledExactlyOnceWith(ITEM, 'add-copy');

    await act(() => userEvent.click(dialog()!.querySelector<HTMLElement>('[role="combobox"]')!));
    await act(() => userEvent.click(dialog()!.querySelector<HTMLElement>('[role="option"][data-value="copy-2"]')!));
    await act(() => button('workflowLibrary.copyChoice.replace')!.click());

    expect(loader.replace).toHaveBeenCalledExactlyOnceWith(ITEM, 'copy-2');
    expect(loader.resume).not.toHaveBeenCalled();
  });

  it('leaves a choice made in another project unasked', async () => {
    await render();
    await act(() => requestLibraryCopyChoice('project-2', ITEM));

    expect(dialog()).toBeNull();
  });

  it('drops the question when its project stops being active', async () => {
    await render();
    await act(() => requestLibraryCopyChoice('project-1', ITEM));
    await vi.waitFor(() => expect(dialog()).not.toBeNull());

    project.id = 'project-2';
    await act(() => project.listeners.forEach((listener) => listener()));

    expect(workflowUiStore.getSnapshot().libraryCopyChoice).toBeNull();
  });
});
