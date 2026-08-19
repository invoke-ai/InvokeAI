import type * as ModelRequirementsModule from '@features/workflow/core/modelRequirements';
import type { WorkflowModelRequirement } from '@features/workflow/core/modelRequirements';
import type { WorkflowLibraryEntry, WorkflowLibraryEntryEnrichment } from '@features/workflow/data/libraryBrowseStore';

import { createProjectGraph } from '@features/workflow/utility';
import { act, useEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useWorkflowLibraryMissingCounts } from './WorkflowRequirementsList';

/**
 * The browse store publishes a fresh `entries` array every time a *single*
 * workflow finishes enriching, with four workers running — so loading a page
 * of 20 re-runs this hook 20 times. Resolution is therefore cached per
 * enrichment object, and this suite is the proof: the real resolver is wrapped
 * in a spy, and the assertion is how many entries each publish re-resolves.
 */
const resolveSpy = vi.hoisted(() => vi.fn());

vi.mock('@features/workflow/core/modelRequirements', async (importOriginal) => {
  const original = await importOriginal<typeof ModelRequirementsModule>();

  return {
    ...original,
    resolveWorkflowModelRequirements: (...args: Parameters<typeof original.resolveWorkflowModelRequirements>) => {
      resolveSpy(...args);

      return original.resolveWorkflowModelRequirements(...args);
    },
  };
});

const models = vi.hoisted(() => ({
  ensureModelsLoaded: vi.fn(() => Promise.resolve()),
  ensureStartersLoaded: vi.fn(),
  installedModels: { current: [] as unknown[] },
}));

vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: models.ensureModelsLoaded,
  ensureStartersLoaded: models.ensureStartersLoaded,
  useActiveInstallSources: () => EMPTY_SOURCES,
  useModelsSelector: (selector: (snapshot: unknown) => unknown) => selector({ models: models.installedModels.current }),
  useStartersSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ response: { starter_models: STARTERS } }),
}));

const EMPTY_SOURCES: ReadonlySet<string> = new Set<string>();
const STARTERS = [
  {
    base: 'flux',
    description: 'FLUX.1 dev',
    is_installed: false,
    name: 'FLUX.1 dev',
    source: 'https://models.test/flux',
    type: 'main',
  },
];

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const FLUX_SLOT: WorkflowModelRequirement = { base: 'flux', kind: 'slot', label: 'FLUX checkpoint', modelType: 'main' };

const readyEnrichment = (): WorkflowLibraryEntryEnrichment => ({
  document: createProjectGraph('requirements-fixture'),
  nodeCount: 1,
  // A new object per call: this is the identity the cache keys on.
  requirements: { primaryBase: 'flux', requirements: [FLUX_SLOT] },
  status: 'ready',
});

const entry = (workflowId: string, enrichment: WorkflowLibraryEntryEnrichment): WorkflowLibraryEntry => ({
  enrichment,
  item: { category: 'user', description: '', name: workflowId, thumbnail_url: null, workflow_id: workflowId },
  tags: [],
});

describe('useWorkflowLibraryMissingCounts', () => {
  let host: HTMLDivElement;
  let root: Root;
  let counts: ReadonlyMap<string, number> | null;

  const Harness = ({ entries }: { entries: readonly WorkflowLibraryEntry[] }) => {
    const result = useWorkflowLibraryMissingCounts(entries);

    // Published after the commit, not during render, so the hook under test is
    // the only thing this component does while rendering.
    useEffect(() => {
      counts = result;
    });

    return null;
  };

  // No StrictMode here: this suite counts resolver invocations, and the
  // deliberate double-render would double every number it asserts.
  const render = async (entries: readonly WorkflowLibraryEntry[]) => {
    await act(() => {
      root.render(<Harness entries={entries} />);
    });
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    counts = null;
    models.installedModels.current = [];
    resolveSpy.mockClear();
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('re-resolves only the entry whose enrichment changed', async () => {
    const stable = entry('wf-a', readyEnrichment());
    const pending = entry('wf-b', { status: 'pending' });

    await render([stable, pending]);

    expect(resolveSpy).toHaveBeenCalledTimes(1);
    expect(counts?.get('wf-a')).toBe(1);
    expect(counts?.has('wf-b')).toBe(false);

    resolveSpy.mockClear();

    // What the store publishes when one more workflow finishes parsing: a new
    // array, a new entry object for the row that changed, and the untouched
    // entry (and its enrichment) carried over by identity.
    await render([stable, entry('wf-b', readyEnrichment())]);

    expect(resolveSpy).toHaveBeenCalledTimes(1);
    expect(counts?.get('wf-a')).toBe(1);
    expect(counts?.get('wf-b')).toBe(1);
  });

  it('re-resolves everything when the installed models change under it', async () => {
    const entries = [entry('wf-a', readyEnrichment()), entry('wf-b', readyEnrichment())];

    await render(entries);

    expect(resolveSpy).toHaveBeenCalledTimes(2);

    resolveSpy.mockClear();
    // The install landed: every row's answer can differ now.
    models.installedModels.current = [{ base: 'flux', hash: 'h', key: 'k', name: 'FLUX.1 dev', type: 'main' }];

    await render([...entries]);

    expect(resolveSpy).toHaveBeenCalledTimes(2);
    expect(counts?.size).toBe(0);
  });
});
