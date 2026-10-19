/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerationUiAdapter } from '@features/generation/react';
import type { Project, ProjectLoadResult } from '@workbench/projectContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { GenerationUiProvider } from '@features/generation/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { GenerateDenoisingStrength } from '@workbench/widgets/canvas/GenerateDenoisingStrength';
import {
  DEFAULT_CANVAS_DENOISING_STRENGTH,
  readCanvasDenoisingStrength,
} from '@workbench/widgets/canvas/invoke/canvasStrength';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createDraftProject } from '@workbench/workbenchState';
import { createWorkbenchStore, type WorkbenchInternalStore } from '@workbench/workbenchStore';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ProjectPushOutcome } from './projectFlush';

import { serializeProjectDocumentV3Json } from './projectDocument';

const harness = vi.hoisted(() => ({
  flushProjectToServer: vi.fn<(project: Project) => Promise<ProjectPushOutcome>>(),
  hydrateProjectFromServer: vi.fn<(projectId: string) => Promise<ProjectLoadResult>>(),
  store: null as unknown as WorkbenchInternalStore,
}));

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => vi.fn(),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: vi.fn() }) }));
vi.mock('./library', () => ({ deleteLibraryProject: vi.fn(), refreshProjectLibrary: vi.fn() }));
// The real aggregate store behind the provider hooks these components read; persistence is the only double.
vi.mock('@workbench/WorkbenchContext', async () => {
  const { useSyncExternalStore } = await import('react');
  return {
    useActiveProjectSelector: <Selected,>(selector: (project: Project) => Selected): Selected =>
      useSyncExternalStore(harness.store.subscribe, () => selector(harness.store.getSnapshot().activeProject)),
    useWorkbenchCommands: () => harness.store.commands,
    useWorkbenchLiveCanvasEngines: () => ({ flushPendingPixels: () => Promise.resolve() }),
    useWorkbenchPersistenceAdapter: () => harness.store.internal.persistence,
    useWorkbenchPersistenceService: () => ({
      flushProjectToServer: harness.flushProjectToServer,
      hydrateProjectFromServer: harness.hydrateProjectFromServer,
      persistEmptySession: () => Promise.resolve(),
      releaseProjectSync: vi.fn(),
    }),
    useWorkbenchQueries: () => harness.store.queries,
  };
});

import { useProjectActions } from './useProjectActions';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
/** What the open and close buttons act on; read when clicked. */
const target = { id: '', name: '' };
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const generationUi = {
  sectionPreferences: { sectionsOpen: {}, setSectionOpen: () => undefined },
} as unknown as GenerationUiAdapter;

const HarnessBody = () => {
  const { closeProject, openProject } = useProjectActions();
  return (
    <ChakraProvider value={system}>
      <button onClick={() => void openProject(target.id, target.name)}>open</button>
      <button onClick={() => closeProject(harness.store.queries.getProject(target.id)!)}>close</button>
      <GenerationUiProvider adapter={generationUi}>
        <GenerateDenoisingStrength>{({ field }) => field}</GenerateDenoisingStrength>
      </GenerationUiProvider>
    </ChakraProvider>
  );
};

// Deleting invalidates the gallery's board lists, so the hook reads the query client.
const queryClient = new QueryClient();
const Harness = () => (
  <QueryClientProvider client={queryClient}>
    <HarnessBody />
  </QueryClientProvider>
);

const click = (label: 'close' | 'open', project: { id: string; name: string }) =>
  act(() => {
    Object.assign(target, { id: project.id, name: project.name });
    [...document.querySelectorAll('button')].find((button) => button.textContent === label)!.click();
  });

const activeProjectId = () => harness.store.getSnapshot().activeProject.id;

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};

const strengthOf = (projectId: string): number =>
  readCanvasDenoisingStrength(getProjectWidgetValues(harness.store.queries.getProject(projectId)!, 'canvas'));

const strengthSlider = (): HTMLElement => {
  const label = [...document.querySelectorAll('[data-scope="scrubber"] [data-part="label"]')].find(
    (candidate) => candidate.textContent === 'widgets.generate.denoisingStrength'
  );
  const slider = label?.closest('[data-scope="scrubber"]')?.querySelector<HTMLElement>('[role="slider"]');
  if (!slider) {
    throw new Error('the strength scrubber did not render');
  }
  return slider;
};

/** One keyboard step: a debounced Generate edit, still held by the component. */
const typeStrength = () =>
  act(() => {
    strengthSlider().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
  });

const typedStrength = DEFAULT_CANVAS_DENOISING_STRENGTH + 0.01;

/** Past the 250 ms draft debounce, so a commit that was still scheduled would have landed by now. */
const outlastDebounce = () =>
  act(async () => {
    await new Promise((resolve) => {
      globalThis.setTimeout(resolve, 300);
    });
  });

const acknowledged = (project: Project): ProjectPushOutcome => ({
  documentJson: serializeProjectDocumentV3Json(project).documentJson,
  kind: 'acknowledged',
});

let originId = '';

beforeEach(async () => {
  accountLifecycle.activate('project-transition-drafts-test');
  harness.store = createWorkbenchStore();
  originId = harness.store.getSnapshot().activeProject.id;
  harness.flushProjectToServer.mockReset();
  harness.flushProjectToServer.mockImplementation((project) => Promise.resolve(acknowledged(project)));
  harness.hydrateProjectFromServer.mockReset();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => root?.render(<Harness />));
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  accountLifecycle.invalidate();
});

describe('project transitions commit pending Generate drafts to their own project', () => {
  it('keeps an edit typed while another project hydrates on the project it was typed in', async () => {
    const hydration = deferred<ProjectLoadResult>();
    harness.hydrateProjectFromServer.mockReturnValueOnce(hydration.promise);
    const opened = createDraftProject(harness.store.getState().projects);

    await click('open', opened);
    await typeStrength();
    // Still only a draft: the transition, not the debounce, must be what commits it.
    expect(strengthOf(originId)).toBe(DEFAULT_CANVAS_DENOISING_STRENGTH);
    await act(async () => {
      hydration.resolve({ project: opened, status: 'loaded' });
      await vi.waitFor(() => expect(activeProjectId()).toBe(opened.id));
    });

    expect(strengthOf(originId)).toBeCloseTo(typedStrength);

    await click('open', { id: originId, name: 'origin' });
    await outlastDebounce();

    expect(activeProjectId()).toBe(originId);
    expect(strengthOf(originId)).toBeCloseTo(typedStrength);
    expect(strengthOf(opened.id)).toBe(DEFAULT_CANVAS_DENOISING_STRENGTH);
    expect(Number(strengthSlider().getAttribute('aria-valuenow'))).toBeCloseTo(typedStrength);
  });

  it('pushes an edit typed while the project is closing before the tab closes', async () => {
    const other = createDraftProject(harness.store.getState().projects);
    harness.store.commands.projects.open(other);
    harness.store.commands.projects.switchTo(originId);
    await act(() => Promise.resolve());
    const firstPush = deferred<void>();
    harness.flushProjectToServer.mockImplementationOnce(async (project) => {
      await firstPush.promise;
      return acknowledged(project);
    });

    await click('close', { id: originId, name: 'origin' });
    await typeStrength();
    expect(strengthOf(originId)).toBe(DEFAULT_CANVAS_DENOISING_STRENGTH);
    await act(async () => {
      firstPush.resolve();
      await vi.waitFor(() => expect(harness.store.queries.getProject(originId)).toBeNull());
    });

    const pushedStrengths = harness.flushProjectToServer.mock.calls.map(([project]) =>
      readCanvasDenoisingStrength(getProjectWidgetValues(project, 'canvas'))
    );
    expect(pushedStrengths.at(-1)).toBeCloseTo(typedStrength);
    expect(pushedStrengths[0]).toBe(DEFAULT_CANVAS_DENOISING_STRENGTH);
    await outlastDebounce();
    expect(strengthOf(other.id)).toBe(DEFAULT_CANVAS_DENOISING_STRENGTH);
  });

  it('keeps an edit typed just before switching to an open project on the project it was typed in', async () => {
    const other = createDraftProject(harness.store.getState().projects);
    harness.store.commands.projects.open(other);
    harness.store.commands.projects.switchTo(originId);
    await act(() => Promise.resolve());

    await typeStrength();
    expect(strengthOf(originId)).toBe(DEFAULT_CANVAS_DENOISING_STRENGTH);
    await click('open', other);
    await outlastDebounce();

    expect(activeProjectId()).toBe(other.id);
    expect(strengthOf(originId)).toBeCloseTo(typedStrength);
    expect(strengthOf(other.id)).toBe(DEFAULT_CANVAS_DENOISING_STRENGTH);
  });

  it('leaves an edit pending when a new project is created (as opening a workflow in one does) on its origin', async () => {
    await typeStrength();
    expect(strengthOf(originId)).toBe(DEFAULT_CANVAS_DENOISING_STRENGTH);
    const created = await act(() => harness.store.commands.projects.create());
    await outlastDebounce();

    expect(harness.store.getSnapshot().activeProject.id).toBe(created.id);
    expect(strengthOf(originId)).toBeCloseTo(typedStrength);
    expect(strengthOf(created.id)).toBe(DEFAULT_CANVAS_DENOISING_STRENGTH);
  });
});
