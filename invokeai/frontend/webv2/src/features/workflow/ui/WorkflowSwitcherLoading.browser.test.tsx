import type { ProjectWorkflowSource } from '@features/workflow/core/types';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { createProjectGraph } from '@features/workflow/utility';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { WorkflowUiProvider } from './WorkflowUiContext';
import { setWorkflowLibraryOpen, workflowUiStore } from './workflowUiStore';
import { WorkflowWidgetLabel } from './WorkflowWidgetChrome';

// The switcher's scrolling list loads on first use. Here it is held back until a test releases it; the menu must open
// whole and work before and as it arrives. A list that fails to load: WorkflowSwitcherLoadFailure.browser.test.tsx.
const scrollableChunk = vi.hoisted(() => {
  let release = () => {};
  const loaded = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { loaded, release: () => release() };
});
vi.mock('@platform/ui/ScrollableMenuContent', async (importOriginal) => {
  await scrollableChunk.loaded;
  return importOriginal();
});
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useAuthSession: () => ({ multiuserEnabled: false, phase: 'ready', user: null }),
}));

const TRANSLATIONS: Record<string, string> = {
  'widgets.workflow.inThisProject': 'In this project',
  'widgets.workflow.libraryWithEllipsis': 'Workflow library…',
  'widgets.workflow.newWorkflow': 'New workflow',
};
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) =>
      key === 'widgets.workflow.switchWorkflowTrigger' ? `Switch ${String(options?.name)}` : (TRANSLATIONS[key] ?? key),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const workflow = (id: string, name: string, source?: ProjectWorkflowSource) => ({
  document: { ...createProjectGraph(id), name },
  source,
});
const WORKFLOWS = [workflow('workflow-1', 'Text 2 Image'), workflow('workflow-2', 'Upscale')];
const SNAPSHOT = {
  activeWorkflow: WORKFLOWS[1]!,
  activeWorkflowId: 'workflow-2',
  galleryValues: {},
  id: 'project-1',
  isWorkflowRunning: false,
  projectGraph: WORKFLOWS[1]!.document,
  workflowValues: {},
  workflows: WORKFLOWS,
};

describe('the workflow switcher while its scrolling list loads', () => {
  let host: HTMLDivElement;
  let root: Root;
  const commands = { createWorkflow: vi.fn(() => 'workflow-3'), selectWorkflow: vi.fn() };

  beforeEach(async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    commands.createWorkflow.mockClear();
    setWorkflowLibraryOpen(false);
    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally minimal test adapter
    const adapter = {
      commands,
      project: { getSnapshot: () => SNAPSHOT, subscribe: () => () => {} },
    } as unknown as WorkflowUiAdapter;
    await act(() =>
      root.render(
        <QueryClientProvider client={new QueryClient()}>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={adapter}>
              <WorkflowWidgetLabel region="center" />
            </WorkflowUiProvider>
          </ChakraProvider>
        </QueryClientProvider>
      )
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    setWorkflowLibraryOpen(false);
  });

  const trigger = () => page.getByRole('button', { name: 'Switch Upscale' });
  const menu = () => document.querySelector<HTMLElement>('[role="menu"]');
  const choices = () => [...document.querySelectorAll('[role="menuitemradio"]')].map((item) => item.textContent);
  const themedList = () => menu()?.querySelector('[data-scope="scroll-area"]') ?? null;
  const library = () => document.querySelector<HTMLElement>('[data-value="workflow-library"]')!;
  const openMenu = async () => {
    await act(() => userEvent.click(trigger()));
    await vi.waitFor(() => expect(menu()).toBeVisible());
    // Measured once the menu's open animation has settled.
    await act(() =>
      Promise.all(
        menu()!
          .getAnimations({ subtree: true })
          .map((animation) => animation.finished)
      )
    );
  };
  const expectLibraryOnThisProject = () =>
    expect(workflowUiStore.getSnapshot()).toMatchObject({
      isLibraryOpen: true,
      librarySelection: { projectId: 'project-1', workflowId: 'workflow-2' },
      libraryTab: 'project',
    });

  it('opens whole before its list loads, keeps its actions in place as the list arrives, and works throughout', async () => {
    await openMenu();
    expect(themedList()).toBeNull();
    expect(choices()).toEqual(['Text 2 Image', 'Upscale']);
    await act(() => userEvent.click(library()));
    expectLibraryOnThisProject();
    await vi.waitFor(() => expect(menu()).toBeNull());

    setWorkflowLibraryOpen(false);
    await openMenu();
    const before = library().getBoundingClientRect().top;
    await act(() => scrollableChunk.release());
    await vi.waitFor(() => expect(themedList()).not.toBeNull(), { timeout: 8000 });
    expect(library().getBoundingClientRect().top).toBe(before);
    expect(menu()!.contains(document.activeElement)).toBe(true);
    await act(() => userEvent.click(library()));
    expectLibraryOnThisProject();
  });
});
