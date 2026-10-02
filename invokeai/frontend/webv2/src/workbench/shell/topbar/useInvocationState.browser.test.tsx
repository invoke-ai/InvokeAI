import type { GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type { Project } from '@workbench/projectContracts';
import type { ReactNode } from 'react';

import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

const harness = vi.hoisted(() => ({ project: null as unknown }));

// A controllable stand-in for the node-template store, so a test can decide which of the two stores
// the route depends on finishes loading last.
const templates = vi.hoisted(() => {
  type Snapshot = { error: string | null; status: string; templates: Record<string, unknown> };
  const loading: Snapshot = { error: null, status: 'loading', templates: {} };
  const listeners = new Set<() => void>();
  let snapshot = loading;

  return {
    get: () => snapshot,
    reset: () => {
      snapshot = loading;
    },
    set: (next: Snapshot) => {
      snapshot = next;
      listeners.forEach((listener) => listener());
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
});

vi.mock('@features/generation/react', () => ({
  useDynamicPrompts: () => ({ count: 1, error: null, isDynamic: false, isError: false, isLoading: false, prompts: [] }),
}));
vi.mock('@features/models', () => ({
  ensureModelsLoaded: () => Promise.resolve(),
  // One snapshot object, as the real store hands out. A fresh `models` array per render would change
  // an input React Compiler memoises the route on, and re-resolve it by accident.
  useModelsSelector: (selector: (snapshot: unknown) => unknown) => selector(modelsSnapshot),
}));
vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getInvocationTemplatesSnapshot: () => templates.get(),
  subscribeInvocationTemplates: templates.subscribe,
}));
vi.mock('@workbench/activeInvocationSubmission', () => ({ submitActiveInvocation: () => Promise.resolve() }));
vi.mock('@workbench/canvasInvocationPreparation', () => ({ useIsCanvasInvocationPreparing: () => false }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(harness.project as Project),
  useWorkbenchCommands: () => ({}),
  useWorkbenchQueries: () => ({}),
  useWorkbenchSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ backendConnection: { status: 'connected' } }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';

import { useInvocationState } from './useInvocationState';

const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl-model', name: 'SDXL', type: 'main' };
const modelsSnapshot = { models: [sdxlModel], status: 'loaded' };

const activeProject = (state: ReturnType<typeof createInitialWorkbenchState>): Project =>
  state.projects.find((candidate) => candidate.id === state.activeProjectId)!;

const buildGenerateProject = (): Project => {
  // The table is needed to derive a *valid* saved project; the point of the test is what happens
  // when the app is restarted and that project is reopened before the table comes back.
  setArchitectureCapabilities(architectureCapabilitiesFixture);
  const values: GenerateWidgetValues = {
    ...getDefaultGenerateSettings(sdxlModel),
    model: sdxlModel,
    modelKey: sdxlModel.key,
    positivePrompt: 'a landscape',
  };
  resetArchitectureCapabilities();

  return activeProject(workbenchReducer(createInitialWorkbenchState(), { type: 'setGenerateSettings', values }));
};

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const Probe = (): ReactNode => {
  const state = useInvocationState();

  return (
    <span data-batch-count={state.batchCount} data-testid="reasons">
      {state.blockingReasons.join(' | ')}
    </span>
  );
};

const renderProbe = async (project: Project) => {
  harness.project = project;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(<Probe />);
  });
};

const blockingReasons = (): string => host?.querySelector('[data-testid="reasons"]')?.textContent ?? '';

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  resetArchitectureCapabilities();
  templates.reset();
});

describe('useInvocationState and the architecture capability table', () => {
  it('stops blocking Invoke as soon as a retried load succeeds', async () => {
    await renderProbe(buildGenerateProject());

    expect(blockingReasons()).toBe(
      'Model capabilities are not available. Generation is blocked until they load; if this persists, retry from the Generate panel.'
    );

    // Capability arrival alone must recompute the route after retry succeeds.
    await act(() => {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    });

    expect(blockingReasons()).toBe('');
  });
});

describe('useInvocationState and the workflow node templates', () => {
  it('re-resolves a workflow route when the templates finish loading after the capability table', async () => {
    // Template arrival must recompute the route even when the capability snapshot is unchanged.
    setArchitectureCapabilities(architectureCapabilitiesFixture);
    await renderProbe(
      activeProject(
        workbenchReducer(createInitialWorkbenchState(), { sourceId: 'workflow', type: 'setInvocationSource' })
      )
    );

    expect(blockingReasons()).toContain('Node definitions are still loading.');

    await act(() => {
      templates.set({ error: null, status: 'loaded', templates: {} });
    });

    expect(blockingReasons()).not.toContain('Node definitions are still loading.');
  });

  it("reports the workflow widget's own iteration count for a workflow route", async () => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
    let state = workbenchReducer(createInitialWorkbenchState(), { sourceId: 'workflow', type: 'setInvocationSource' });
    state = workbenchReducer(state, { type: 'patchWidgetValues', values: { batchCount: 5 }, widgetId: 'generate' });
    state = workbenchReducer(state, { type: 'patchWidgetValues', values: { batchCount: 3 }, widgetId: 'workflow' });
    await renderProbe(activeProject(state));

    expect(host?.querySelector('[data-testid="reasons"]')?.getAttribute('data-batch-count')).toBe('3');
  });
});
