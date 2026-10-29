import type { GenerateWidgetValues, GenerationIssue, MainModelConfig } from '@features/generation/contracts';
import type * as IdentityModule from '@features/identity';
import type * as ProjectsApiModule from '@workbench/projects/api';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { DEFAULT_PREFERENCES, getWorkbenchPreferences, patchWorkbenchPreferences } from '@workbench/settings/store';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { WorkbenchWidgetRegistryProvider } from '@workbench/WorkbenchWidgetRegistryContext';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { InvokeIssueAction } from './invokeIssueActions';
import type { InvocationIssue } from './invokeIssueTargets';

const workbench = vi.hoisted(() => ({
  canManageModels: true,
  store: null as ReturnType<typeof createWorkbenchStore> | null,
}));

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ProjectsApiModule>()),
  setClientStateValue: async () => {},
}));
// A real store without the provider's persistence: every action acts on it.
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useWorkbenchCommands: () => workbench.store!.commands,
  useWorkbenchQueries: () => workbench.store!.queries,
}));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<typeof IdentityModule>()),
  useCapabilities: () => ({ canManageModels: workbench.canManageModels }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

const { useInvokeIssueActions } = await import('./invokeIssueActions');

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl-model', name: 'SDXL', type: 'main' };

let host: HTMLDivElement;
let root: Root;
let getActions: ((issue: InvocationIssue) => InvokeIssueAction[]) | null = null;

/** Publishes the hook's latest result on every render, so a re-render with other capabilities is seen. */
const Probe = ({ onActions }: { onActions: (actions: (issue: InvocationIssue) => InvokeIssueAction[]) => void }) => {
  onActions(useInvokeIssueActions());

  return null;
};
const publish = (actions: (issue: InvocationIssue) => InvokeIssueAction[]) => {
  getActions = actions;
};
const renderProbe = () =>
  act(() =>
    root.render(
      <WorkbenchWidgetRegistryProvider getWidgetById={getWidgetById} getWidgetsForRegion={getWidgetsForRegion}>
        <Probe onActions={publish} />
      </WorkbenchWidgetRegistryProvider>
    )
  );

const store = () => workbench.store!;
const project = () => store().getSnapshot().activeProject;
const isPlaced = (typeId: string) =>
  Object.values(project().widgetRegions).some((region) =>
    region.instanceIds.some((instanceId) => project().widgetInstances[instanceId]?.typeId === typeId)
  );
const actionsFor = (issue: InvocationIssue) => getActions!(issue);
const run = async (issue: InvocationIssue, id: string) => {
  const action = actionsFor(issue).find((candidate) => candidate.id === id);

  expect(action).toBeDefined();
  await act(() => action!.run());
};

const generationIssue = (overrides: Partial<GenerationIssue> = {}): GenerationIssue => ({
  id: 'guidance:cfgScale:range',
  message: 'Guidance is out of range.',
  target: { field: 'cfgScale', section: 'guidance' },
  ...overrides,
});

beforeEach(async () => {
  accountLifecycle.activate('invoke-issue-actions-test');
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  setArchitectureCapabilities(architectureCapabilitiesFixture);
  const values: GenerateWidgetValues = {
    ...getDefaultGenerateSettings(sdxlModel),
    cfgScale: 30,
    model: sdxlModel,
    modelKey: sdxlModel.key,
  };
  resetArchitectureCapabilities();
  workbench.store = createWorkbenchStore();
  store().commands.layout.applyPreset('compose');
  store().commands.generation.setSettings(values);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await renderProbe();
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  getActions = null;
  workbench.store = null;
  accountLifecycle.invalidate();
});

describe('useInvokeIssueActions', () => {
  it('offers nothing for a lost connection: it clears on reconnect', () => {
    expect(actionsFor({ message: 'The server is disconnected.', target: { kind: 'connection' } })).toEqual([]);
  });

  it('applies a Generate fix in place and keeps the list open', async () => {
    const issue: InvocationIssue = {
      message: 'Guidance is out of range.',
      target: { issue: generationIssue({ fix: { kind: 'clampGuidance', value: 7 } }), kind: 'generation' },
    };

    expect(actionsFor(issue).find(({ id }) => id === 'fix')?.stays).toBe(true);
    await run(issue, 'fix');

    expect(getProjectWidgetValues(project(), 'generate').cfgScale).toBe(7);
  });

  it('shows a Generate issue by opening its section and the Generate panel', async () => {
    await patchWorkbenchPreferences({ generateSectionsOpen: { guidance: false } });
    await run(
      { message: 'Guidance is out of range.', target: { issue: generationIssue(), kind: 'generation' } },
      'show'
    );

    expect(getWorkbenchPreferences().generateSectionsOpen.guidance).toBe(true);
    expect(isPlaced('generate')).toBe(true);
  });

  it('sends a missing model to the Model Manager, filtered to main models, for those who can install', async () => {
    const originalHash = window.location.hash;
    const issue: InvocationIssue = {
      message: 'SDXL is not installed.',
      target: {
        issue: generationIssue({ id: 'model:notInstalled:sdxl-model', target: { section: 'model' } }),
        kind: 'generation',
      },
    };

    workbench.canManageModels = false;
    await renderProbe();
    expect(actionsFor(issue).map(({ id }) => id)).toEqual(['show']);
    workbench.canManageModels = true;
    await renderProbe();
    await run(issue, 'models');

    await expect.poll(() => window.location.hash).toBe(`#/models?project=${encodeURIComponent(project().id)}`);
    window.location.hash = originalHash;
  });

  it('opens the workflow editor in the center to show a node', async () => {
    await run({ message: 'Missing input.', target: { kind: 'workflowNode', nodeId: 'node-1' } }, 'show');

    const center = project().widgetRegions.center;
    expect(project().widgetInstances[center.activeInstanceId]?.typeId).toBe('workflow');
  });

  it('adds a source widget the layout lacks', async () => {
    expect(isPlaced('video')).toBe(false);

    await run({ message: 'Not in this layout.', target: { kind: 'mountSource', sourceId: 'video' } }, 'add');

    expect(isPlaced('video')).toBe(true);
  });

  it('switches a video route to the Gallery in place', async () => {
    store().commands.generation.setDestination('canvas');
    const issue: InvocationIssue = {
      message: 'Video goes to the Gallery.',
      target: { destination: 'gallery', kind: 'destination' },
    };

    expect(actionsFor(issue)[0]?.stays).toBe(true);
    await run(issue, 'destination');

    expect(project().invocation.destination).toBe('gallery');
  });

  it('opens the canvas and its Layers for a layer issue, and the source widget for the rest', async () => {
    await run({ message: 'Choose a model.', target: { kind: 'layer', layerId: 'control-1' } }, 'show');
    expect(isPlaced('layers')).toBe(true);

    await run({ message: 'Settings are incomplete.', target: { kind: 'widget', widgetId: 'video' } }, 'show');
    expect(isPlaced('video')).toBe(true);

    await run({ message: 'The prompt could not be expanded.', target: { kind: 'prompt' } }, 'show');
    expect(isPlaced('generate')).toBe(true);
  });
});
