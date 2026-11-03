import type { GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type { CanvasControlLayerContract } from '@workbench/canvas-engine/api';
import type { Project } from '@workbench/projectContracts';
import type * as settingsStoreModule from '@workbench/settings/store';
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

const harness = vi.hoisted(() => ({
  project: null as unknown,
  submit: vi.fn((_args: Record<string, unknown>) => Promise.resolve()),
}));

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

// A controllable stand-in for the account preferences, so a test can change the remembered control-model picks.
const preferences = vi.hoisted(() => {
  const listeners = new Set<() => void>();
  let controlModelMemory: Record<string, string> = {};

  return {
    get: () => controlModelMemory,
    reset: () => {
      controlModelMemory = {};
    },
    set: (next: Record<string, string>) => {
      controlModelMemory = next;
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
vi.mock('@workbench/settings/store', async (importOriginal) => {
  const actual = await importOriginal<typeof settingsStoreModule>();
  return {
    ...actual,
    getWorkbenchPreferences: () => ({ ...actual.getWorkbenchPreferences(), controlModelMemory: preferences.get() }),
    subscribeWorkbenchPreferences: (listener: () => void) => preferences.subscribe(listener),
  };
});
vi.mock('@workbench/activeInvocationSubmission', () => ({ submitActiveInvocation: harness.submit }));
vi.mock('@workbench/invocationPreparation', () => ({ useIsInvocationPreparing: () => false }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(harness.project as Project),
  useWorkbenchCommands: () => ({}),
  useWorkbenchQueries: () => ({}),
  useWorkbenchSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ backendConnection: { status: 'connected' } }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';

import { useInvocationState } from './useInvocationState';

const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl-model', name: 'SDXL', type: 'main' };
const depthModel = (key: string) => ({ base: 'sdxl', key, name: key, type: 'controlnet' });
const modelsSnapshot = {
  models: [sdxlModel, depthModel('sdxl-depth-a'), depthModel('sdxl-depth-b')],
  status: 'loaded',
};

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
    <>
      <span data-batch-count={state.batchCount} data-testid="reasons">
        {state.issues.map((issue) => issue.message).join(' | ')}
      </span>
      <span data-testid="issue-layers">
        {state.issues.map(({ target }) => (target.kind === 'layer' ? target.layerId : '-')).join(' | ')}
      </span>
      <button data-testid="invoke-front" type="button" onClick={() => void state.invoke({ prepend: true })}>
        Invoke front
      </button>
      <button data-testid="invoke" type="button" onClick={() => void state.invoke()}>
        Invoke
      </button>
    </>
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
  preferences.reset();
});

describe('useInvocationState and the architecture capability table', () => {
  it('stops blocking Invoke as soon as a retried load succeeds', async () => {
    await renderProbe(buildGenerateProject());

    // Worded from the locale, which this probe leaves untranslated.
    expect(blockingReasons()).toBe('widgets.generate.issues.capabilitiesUnavailable');

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

    expect(blockingReasons()).toContain('nodes.readiness.templatesLoading');

    await act(() => {
      templates.set({ error: null, status: 'loaded', templates: {} });
    });

    expect(blockingReasons()).not.toContain('nodes.readiness.templatesLoading');
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

describe('useInvocationState invoke', () => {
  it('forwards a front-of-queue request to the submission, and leaves it out otherwise', async () => {
    harness.submit.mockClear();
    await renderProbe(buildGenerateProject());

    await act(() => host!.querySelector<HTMLButtonElement>('[data-testid="invoke-front"]')!.click());
    await act(() => host!.querySelector<HTMLButtonElement>('[data-testid="invoke"]')!.click());

    expect(harness.submit.mock.calls[0]![0]).toMatchObject({ prepend: true });
    expect(harness.submit.mock.calls[1]![0].prepend).toBeUndefined();
  });
});

describe('useInvocationState and the remembered control-model picks', () => {
  it('re-resolves a canvas route when a remembered pick settles a choice between control models', async () => {
    const depthLayer: CanvasControlLayerContract = {
      adapter: { beginEndStepPct: [0, 0.75], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 0.75 },
      blendMode: 'normal',
      controlType: 'depth',
      detection: null,
      id: 'depth-layer',
      isEnabled: true,
      isLocked: false,
      name: 'Depth 1',
      opacity: 1,
      source: { image: { height: 64, imageName: 'depth.png', width: 64 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: true,
    };
    const project = buildGenerateProject();
    setArchitectureCapabilities(architectureCapabilitiesFixture);
    await renderProbe({
      ...project,
      canvas: { ...project.canvas, document: { ...project.canvas.document, stacks: stacksFrom([depthLayer]) } },
      invocation: { ...project.invocation, sourceId: 'canvas' },
    });

    expect(blockingReasons()).toContain('widgets.layers.control.invalidLayer');
    // The control layer's issue names the layer, so Invoke can show it; the others name none.
    const layers = host!.querySelector('[data-testid="issue-layers"]')!.textContent!.split(' | ');
    expect(layers.filter((layerId) => layerId !== '-')).toEqual(['depth-layer']);
    expect(blockingReasons().split(' | ')[layers.indexOf('depth-layer')]).toContain(
      'widgets.layers.control.invalidLayer'
    );

    await act(() => {
      preferences.set({ 'sdxl:depth': 'sdxl-depth-b' });
    });

    expect(blockingReasons()).not.toContain('widgets.layers.control.invalidLayer');
  });
});
