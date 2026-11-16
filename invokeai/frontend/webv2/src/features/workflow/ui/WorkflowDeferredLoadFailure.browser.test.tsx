/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop -- explicit per-case test props */
import type * as DeferredModuleModule from '@platform/react/deferredModule';

import { ChakraProvider } from '@chakra-ui/react';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, type ComponentProps, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { WorkflowUiAdapter } from './WorkflowUiContext';

import { WorkflowPreviewJson } from './graph-preview/WorkflowPreviewStage';
import { DeferredGraphPreviewSnapshot, DeferredWorkflowDocumentGraph } from './library/libraryPreviewSource';
import { WorkflowUiProvider } from './WorkflowUiContext';
import { requestWorkflowRename, setWorkflowLibraryOpen, workflowUiStore } from './workflowUiStore';
import { WorkflowDialogHost } from './WorkflowWidgetChrome';

const chunk = vi.hoisted(() => ({ fails: true, rejections: 0 }));
// Fail the actual deferred import seam, rather than a Vitest module response (which would never settle its request).
vi.mock('@platform/react/deferredModule', async (importOriginal) => {
  const actual = await importOriginal<typeof DeferredModuleModule>();
  return {
    ...actual,
    createDeferredModule: <Module,>(load: () => Promise<Module>) =>
      actual.createDeferredModule(() => {
        if (chunk.fails) {
          chunk.rejections += 1;
          return Promise.reject(new TypeError('Failed to fetch dynamically imported module'));
        }
        return load();
      }),
  };
});
vi.mock('./editor/AddNodeDialog', () => ({ AddNodeDialog: () => null }));
vi.mock('./editor/CallSavedWorkflowSyncRuntime', () => ({ CallSavedWorkflowSyncRuntime: () => null }));
vi.mock('./library/WorkflowPublicationHost', () => ({ WorkflowPublicationHost: () => null }));
vi.mock('./PendingLibraryWorkflowLoader', () => ({ PendingWorkflowLoader: () => null }));
vi.mock('./library/WorkflowLibraryDialog', () => ({
  WorkflowLibraryDialog: ({ isOpen }: { isOpen: boolean }) => (isOpen ? <p>loaded library</p> : null),
}));
vi.mock('./graph-preview/WorkflowPreviewJson', () => ({ default: () => <p>loaded JSON</p> }));
vi.mock('./graph-preview/GraphPreviewDialog', () => ({ WorkflowDocumentGraph: () => <p>loaded graph</p> }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it('keeps failed library and preview imports local and recovers them without reloading the page', async () => {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  const workflow = { document: createProjectGraph('workflow') };
  const project = {
    activeWorkflow: workflow,
    activeWorkflowId: 'workflow',
    galleryValues: {},
    id: 'project',
    isWorkflowRunning: false,
    projectGraph: workflow.document,
    workflows: [workflow],
    workflowValues: {},
  };
  const adapter = {
    commands: { renameWorkflow: vi.fn() },
    project: { getSnapshot: () => project, subscribe: () => () => {} },
    notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
  } as unknown as WorkflowUiAdapter;
  const render = (children: React.ReactNode) =>
    act(() =>
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={adapter}>{children}</WorkflowUiProvider>
          </ChakraProvider>
        </StrictMode>
      )
    );
  const retry = () => act(() => userEvent.click(page.getByRole('button', { name: 'common.retry' })));

  try {
    setWorkflowLibraryOpen(true);
    await render(<WorkflowDialogHost />);
    await vi.waitFor(() =>
      expect(document.querySelector('[role="alert"]')?.textContent).toContain('workflowLibrary.dialogLoadFailed')
    );
    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => expect(workflowUiStore.getSnapshot().isLibraryOpen).toBe(false));
    // The import failure must leave the rest of this host usable, rather than trip the widget's boundary.
    await act(() => requestWorkflowRename('workflow'));
    await expect.element(page.getByRole('textbox')).toBeVisible();
    await act(() => userEvent.keyboard('{Escape}'));
    await act(() => setWorkflowLibraryOpen(true));
    chunk.fails = false;
    await retry();
    await vi.waitFor(() => expect(host.textContent).toContain('loaded library'));
    await act(() => setWorkflowLibraryOpen(false));

    chunk.fails = true;
    await render(
      <WorkflowPreviewJson copyFailedLabel="failed" copyLabel="copy" getText={() => '{}'} label="Graph JSON" />
    );
    await vi.waitFor(() =>
      expect(document.querySelector('[role="alert"]')?.textContent).toContain('workflowLibrary.preview.loadFailed')
    );
    chunk.fails = false;
    await retry();
    await vi.waitFor(() => expect(host.textContent).toContain('loaded JSON'));

    chunk.fails = true;
    const onError = vi.fn();
    await render(
      <DeferredGraphPreviewSnapshot
        graph={{ edges: [], id: 'graph', nodes: [], version: 1 }}
        onCapture={vi.fn()}
        onError={onError}
      />
    );
    await vi.waitFor(() => expect(onError).toHaveBeenCalled());
    expect(onError.mock.calls[0]![0]).toBeInstanceOf(TypeError);
    await render(null);
    const props = {} as ComponentProps<typeof DeferredWorkflowDocumentGraph>;
    await render(<DeferredWorkflowDocumentGraph {...props} />);
    await vi.waitFor(() =>
      expect(document.querySelector('[role="alert"]')?.textContent).toContain('workflowLibrary.preview.loadFailed')
    );
    chunk.fails = false;
    await retry();
    await vi.waitFor(() => expect(host.textContent).toContain('loaded graph'));
  } finally {
    await act(() => root.unmount());
    host.remove();
    setWorkflowLibraryOpen(false);
    workflowUiStore.patchSnapshot({ renameRequest: null });
  }
});
