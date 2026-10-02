import type { ImageMapState } from '@workbench/image-map/api';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Stand-ins for the gallery's values and the workbench commands the chip and Esc write through.
const workbench = vi.hoisted(() => ({
  galleryValues: {} as Record<string, unknown>,
  patchValues: vi.fn(),
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  // The gallery's values drive the cluster chip and the Esc clear. Every other widget's selector is applied to an
  // empty value bag rather than answering `false`: each accessor owns its own default, and handing a boolean to a
  // setting that is a number or null makes the widget act on a value it could never be given in production.
  useWidgetValuesSelector: (widgetId: string, select: (values: Record<string, unknown>) => unknown) =>
    select(widgetId === 'gallery' ? workbench.galleryValues : {}),
  useWorkbenchCommands: () => ({ widgets: { patchValues: workbench.patchValues } }),
  useWorkbenchQueries: () => ({ getSnapshot: () => ({ activeProject: {} }) }),
}));
vi.mock('@workbench/widgetState', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProjectWidgetValues: () => workbench.galleryValues,
}));

// Stub the heavy Plotly sibling; this test owns the progress badge.
vi.mock('./ImageMapPlot', () => ({ default: () => <div data-testid="plot" /> }));

const models = vi.hoisted(() => {
  // A real subscribable store, like the one behind `useActiveInstallSources`:
  // putting the encoder into (and out of) an in-flight install has to re-render
  // whoever read it, which is exactly the path the widget depends on.
  const listeners = new Set<() => void>();
  let active: ReadonlySet<string> = new Set<string>();

  return {
    activeSources: {
      get: (): ReadonlySet<string> => active,
      set: (next: Iterable<string>): void => {
        active = new Set(next);
        for (const listener of listeners) {
          listener();
        }
      },
      subscribe: (listener: () => void): (() => void) => {
        listeners.add(listener);

        return () => listeners.delete(listener);
      },
    },
    ensureStartersLoaded: vi.fn(),
    install: vi.fn((_request: { config?: unknown; source: string }) => Promise.resolve(true)),
  };
});

vi.mock('@features/models', async (importOriginal) => {
  const { useSyncExternalStore } = await import('react');

  return {
    ...(await importOriginal<Record<string, unknown>>()),
    ensureStartersLoaded: models.ensureStartersLoaded,
    useActiveInstallSources: () =>
      useSyncExternalStore(models.activeSources.subscribe, models.activeSources.get, models.activeSources.get),
    useInstallActions: () => ({ install: models.install, installMany: vi.fn(), pendingSources: EMPTY_SOURCES }),
    useStartersSelector: (selector: (snapshot: unknown) => unknown) =>
      selector({ response: { starter_models: STARTERS } }),
  };
});

const EMPTY_SOURCES: ReadonlySet<string> = new Set<string>();
const ENCODER_SOURCE = 'apple/DFN2B-CLIP-ViT-L-14-39B';
const ENCODER_DEPENDENCY_SOURCE = 'InvokeAI/encoder-preprocessor';
const STARTERS = [
  {
    base: 'any',
    dependencies: [
      {
        base: 'any',
        description: 'Preprocessor',
        is_installed: false,
        name: 'encoder-preprocessor',
        source: ENCODER_DEPENDENCY_SOURCE,
        type: 'clip_vision',
      },
    ],
    description: 'DFN2B CLIP ViT-L Image Encoder',
    is_installed: false,
    name: 'DFN2B-CLIP-ViT-L-14-39B',
    source: ENCODER_SOURCE,
    type: 'clip_vision',
  },
];

vi.mock('@workbench/image-map/imageMapStore', async (importOriginal) => {
  const original = (await importOriginal()) as object;

  return {
    ...original,
    ensureImageMapLoaded: vi.fn(),
    refreshImageIndexStatus: vi.fn(),
    refreshImageMapPoints: vi.fn(),
    setClusterLabelsEnabled: vi.fn(),
  };
});

import { registerImageCluster } from '@features/gallery/contracts';
import { imageMapStore, refreshImageIndexStatus, refreshImageMapPoints } from '@workbench/image-map/imageMapStore';

import { ImageMapWidgetView } from './ImageMapWidgetView';

let host: HTMLDivElement | null = null;
let root: Root | null = null;

/** Captures what the view registers, and what it disposes, so the Esc binding can be asserted and fired. */
const registered = {
  commands: [] as { handler: () => unknown; id: string; runtime: string; disposed: boolean }[],
  hotkeys: [] as { commandId: string; defaultKeys: string[]; runtime: string; disposed: boolean }[],
};
const makeViewProps = (runtimeName: string) =>
  ({
    runtime: {
      commands: {
        register: (command: { handler: () => unknown; id: string }) => {
          const entry = { ...command, disposed: false, runtime: runtimeName };
          registered.commands.push(entry);
          return () => {
            entry.disposed = true;
          };
        },
      },
      hotkeys: {
        register: (hotkey: { commandId: string; defaultKeys: string[] }) => {
          const entry = { ...hotkey, disposed: false, runtime: runtimeName };
          registered.hotkeys.push(entry);
          return () => {
            entry.disposed = true;
          };
        },
      },
    },
  }) as unknown as WidgetViewProps;
const viewProps = makeViewProps('project-a');
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const dataFor = (
  state: Extract<ImageMapState, 'disabled' | 'model_missing'>,
  modelName = 'clip-vit-large-patch14'
) => ({
  clusterEps: null,
  modelName: state === 'model_missing' ? modelName : null,
  pointCount: 0,
  points: [],
  stale: false,
  state,
  updatedAt: null,
  visibleHash: null,
});

const renderState = async (state: Extract<ImageMapState, 'disabled' | 'model_missing'>, modelName?: string) => {
  imageMapStore.setSnapshot({
    clusterLabels: null,
    clusterLabelsEps: null,
    clusterLabelsHash: null,
    data: dataFor(state, modelName),
    error: null,
    indexCounts: null,
    indexUpdatedAt: null,
    loadState: 'loaded',
    renderError: null,
  });

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <ImageMapWidgetView {...viewProps} />
      </ChakraProvider>
    )
  );
};

/** Every button the message currently offers, in DOM order. */
const buttonLabels = (): string[] =>
  Array.from(host?.querySelectorAll('button') ?? []).map((button) => button.textContent ?? '');

/** The install jobs the app currently has in flight, as the widget sees them. */
const setActiveInstalls = async (sources: string[]) => {
  await act(() => {
    models.activeSources.set(sources);
  });
};

beforeEach(() => {
  vi.clearAllMocks();
  workbench.galleryValues = {};
  registered.commands = [];
  registered.hotkeys = [];
  models.activeSources.set([]);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('Image Map unavailable states', () => {
  it.each([
    [
      'model_missing',
      'Embedding model not installed',
      'To enable image indexing, install the image encoder model',
      // A standing action: nothing polls while the indexer is inert, so the
      // message has to offer its own way to ask the server again.
      ['Check again'],
    ],
    [
      'disabled',
      'Image indexing is off',
      'To enable image indexing, set `image_index_enabled: true` in the server configuration file and restart the server to build a semantic index of your gallery.',
      [],
    ],
  ] as const)(
    'keeps the %s diagnosis visible while exposing a failed refresh',
    async (state, title, detail, actions) => {
      await renderState(state);

      expect(host?.textContent).toContain(title);
      expect(host?.textContent).toContain(detail);
      expect(buttonLabels()).toEqual(actions);

      await act(() => {
        imageMapStore.patchSnapshot({ error: 'The server could not be reached.', loadState: 'error' });
      });

      expect(host?.textContent).toContain(title);
      expect(host?.textContent).toContain('The server could not be reached.');
      expect(host?.querySelector('[role="alert"]')?.textContent).toBe('The server could not be reached.');
      expect(buttonLabels()).toEqual(['Retry']);
    }
  );
});

describe('Image Map missing-model install link', () => {
  it('queues the starter install for the configured encoder', async () => {
    await renderState('model_missing', 'DFN2B-CLIP-ViT-L-14-39B');

    expect(host?.textContent).toContain(
      'To enable image indexing, install the image encoder model DFN2B-CLIP-ViT-L-14-39B from the Model Manager to build a semantic index of your gallery.'
    );

    const link = host?.querySelector('button');

    expect(link?.textContent).toBe('DFN2B-CLIP-ViT-L-14-39B');

    await act(async () => {
      link?.click();
      await Promise.resolve();
    });

    expect(models.install).toHaveBeenCalledWith(expect.objectContaining({ source: 'apple/DFN2B-CLIP-ViT-L-14-39B' }));
  });

  it('queues the starter dependencies ahead of the model itself', async () => {
    await renderState('model_missing', 'DFN2B-CLIP-ViT-L-14-39B');

    await act(async () => {
      host?.querySelector('button')?.click();
      await Promise.resolve();
    });

    expect(models.install.mock.calls.map(([request]) => request.source)).toEqual([
      ENCODER_DEPENDENCY_SOURCE,
      ENCODER_SOURCE,
    ]);
    // The curated metadata rides along, so the model registers under its
    // starter identity instead of whatever probing guesses.
    expect(models.install).toHaveBeenLastCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({ name: 'DFN2B-CLIP-ViT-L-14-39B', type: 'clip_vision' }),
        source: ENCODER_SOURCE,
      })
    );
  });

  it('reports an in-flight download instead of offering the install again', async () => {
    models.activeSources.set([ENCODER_SOURCE]);
    await renderState('model_missing', 'DFN2B-CLIP-ViT-L-14-39B');

    expect(host?.textContent).toContain('DFN2B-CLIP-ViT-L-14-39B (installing…)');
    expect(buttonLabels()).toEqual(['Check again']);
  });

  it('asks the server again as soon as the download lands', async () => {
    models.activeSources.set([ENCODER_SOURCE]);
    await renderState('model_missing', 'DFN2B-CLIP-ViT-L-14-39B');

    expect(refreshImageMapPoints).not.toHaveBeenCalled();

    await setActiveInstalls([]);

    // The server picks the encoder up on the next request, so this refresh is
    // what starts indexing without a restart.
    expect(refreshImageMapPoints).toHaveBeenCalledTimes(1);
    expect(refreshImageIndexStatus).toHaveBeenCalledTimes(1);
  });

  it('keeps the message and the in-flight install mounted while a refresh runs', async () => {
    models.activeSources.set([ENCODER_SOURCE]);
    await renderState('model_missing', 'DFN2B-CLIP-ViT-L-14-39B');

    // A refresh flips loadState to `loading`; a spinner here would unmount the
    // link and lose both the download's pending state and the watcher that
    // refreshes when it lands.
    await act(() => {
      imageMapStore.patchSnapshot({ loadState: 'loading' });
    });

    expect(host?.textContent).toContain('DFN2B-CLIP-ViT-L-14-39B (installing…)');

    await setActiveInstalls([]);

    expect(refreshImageMapPoints).toHaveBeenCalledTimes(1);
  });

  it('falls back to the default encoder name when the server reports none', async () => {
    await renderState('model_missing', '');

    expect(host?.textContent).toContain(
      'install the image encoder model DFN2B-CLIP-ViT-L-14-39B from the Model Manager'
    );
  });

  it('leaves a model the starter catalog does not carry as plain text', async () => {
    await renderState('model_missing');

    expect(host?.textContent).toContain(
      'install the image encoder model clip-vit-large-patch14 from the Model Manager'
    );
    expect(buttonLabels()).toEqual(['Check again']);
  });
});

describe('Image Map indexing activity', () => {
  const renderMapWithCounts = async (
    counts: { total: number; embedded: number; pending: number; failed: number } | null
  ) => {
    imageMapStore.setSnapshot({
      clusterLabels: null,
      clusterLabelsEps: null,
      clusterLabelsHash: null,
      data: {
        clusterEps: null,
        modelName: null,
        pointCount: 2,
        points: [
          { cluster: 0, item: { kind: 'image', name: 'a.png' }, key: 'image:a.png', x: 0, y: 0 },
          { cluster: 0, item: { kind: 'image', name: 'b.png' }, key: 'image:b.png', x: 1, y: 1 },
        ],
        stale: false,
        state: 'ready',
        updatedAt: '2026-08-24T01:00:00',
        visibleHash: 'hash',
      },
      error: null,
      indexCounts: counts,
      indexUpdatedAt: counts ? Date.now() : null,
      loadState: 'loaded',
      renderError: null,
    });

    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <ImageMapWidgetView {...viewProps} />
        </ChakraProvider>
      )
    );
    // Poll for the resolved lazy plot tree rather than assuming a fixed number of Suspense flushes.
    for (let attempt = 0; attempt < 50 && !host?.querySelector('[data-testid="plot"]'); attempt += 1) {
      await act(async () => {
        await new Promise((resolve) => {
          setTimeout(resolve, 10);
        });
      });
    }
  };

  it('reports an index run over the map instead of drawing it silently', async () => {
    // Keep a usable stale map during reindexing but show progress so missing labels are explained.
    await renderMapWithCounts({ embedded: 1204, failed: 0, pending: 16846, total: 18050 });

    expect(host?.querySelector('[data-testid="plot"]')).not.toBeNull();
    expect(host?.textContent).toContain('indexing 1,204/18,050');
    // The map stays: the badge must not replace it.
    expect(host?.textContent).not.toContain('Indexing gallery');
  });

  it('names the labels in the badge, since they vanish while the vocabulary rebuilds', async () => {
    await renderMapWithCounts({ embedded: 1204, failed: 0, pending: 16846, total: 18050 });

    const progressbar = host?.querySelector('[role="progressbar"]');

    expect(progressbar?.getAttribute('aria-label')).toContain('cluster labels update as images finish');
  });

  it('shows no badge once the index is idle', async () => {
    await renderMapWithCounts({ embedded: 18050, failed: 0, pending: 0, total: 18050 });

    expect(host?.querySelector('[data-testid="plot"]')).not.toBeNull();
    expect(host?.textContent).not.toContain('indexing');
  });

  it('shows no badge when the counts are absent, as for a non-admin', async () => {
    await renderMapWithCounts(null);

    expect(host?.querySelector('[data-testid="plot"]')).not.toBeNull();
    expect(host?.textContent).not.toContain('indexing');
  });
});

describe('Image Map cluster selection chip', () => {
  const POINTS = [
    { cluster: 2, item: { kind: 'image' as const, name: 'a.png' }, key: 'image:a.png' as const, x: 0, y: 0 },
    { cluster: 2, item: { kind: 'image' as const, name: 'b.png' }, key: 'image:b.png' as const, x: 1, y: 1 },
    { cluster: 5, item: { kind: 'image' as const, name: 'c.png' }, key: 'image:c.png' as const, x: 9, y: 9 },
  ];

  const renderMap = async (
    counts: { total: number; embedded: number; pending: number; failed: number } | null,
    props: WidgetViewProps = viewProps,
    centerChromeInset?: string
  ) => {
    imageMapStore.setSnapshot({
      clusterLabels: null,
      clusterLabelsEps: null,
      clusterLabelsHash: null,
      data: {
        clusterEps: null,
        modelName: null,
        pointCount: POINTS.length,
        points: POINTS,
        stale: false,
        state: 'ready',
        updatedAt: '2026-09-23T01:00:00',
        visibleHash: 'hash',
      },
      error: null,
      indexCounts: counts,
      indexUpdatedAt: counts ? Date.now() : null,
      loadState: 'loaded',
      renderError: null,
    });

    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <div
            style={{
              height: 400,
              width: 500,
              ...(centerChromeInset ? { '--wb-center-chrome-inset': centerChromeInset } : {}),
            }}
          >
            <ImageMapWidgetView {...props} />
          </div>
        </ChakraProvider>
      )
    );
  };

  const selectCluster = (label: string, count = 2) => {
    const keys = POINTS.slice(0, count).map((point) => point.key);
    const clusterId = registerImageCluster(keys, label);

    workbench.galleryValues = { semanticImageQuery: { clusterId, kind: 'cluster', label }, searchTerm: '' };
  };

  const chip = () => host?.querySelector<HTMLElement>('[role="group"][aria-label="Selected cluster"]') ?? null;
  const CLEARED = { galleryPage: 0, searchTerm: '', semanticImageQuery: null, semanticSearchText: null };

  it('names the selected cluster with its size and colour, above the indexing badge', async () => {
    selectCluster('sunset beach');
    await renderMap({ embedded: 10, failed: 0, pending: 5, total: 15 });

    expect(chip()?.textContent).toBe('sunset beach· 2 items');
    // Cluster 2's palette colour: both members are in it.
    expect(getComputedStyle(chip()!.querySelector('[aria-hidden="true"]')!).backgroundColor).toBe('rgb(225, 87, 89)');
    // Stacked in one corner rather than drawn over each other, the selection first.
    const badge = host?.querySelector('[role="progressbar"]');
    expect(chip()!.compareDocumentPosition(badge!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(badge!.getBoundingClientRect().top).toBeGreaterThanOrEqual(chip()!.getBoundingClientRect().bottom);
  });

  it('shows only the current size for an unlabeled cluster, even after members were deleted', async () => {
    // Named "2 items" at click time; one member has since gone.
    selectCluster('2 items', 1);
    await renderMap(null);

    expect(chip()?.textContent).toBe('1 item');
  });

  it('starts below the floating view selector when the map is in the center area', async () => {
    selectCluster('sunset beach');
    await renderMap(null, viewProps, '3rem');

    const surface = host!.querySelector<HTMLElement>('[data-image-map-surface]')!;
    // 3rem of center chrome plus the corner's own spacing.
    expect(chip()!.getBoundingClientRect().top - surface.getBoundingClientRect().top).toBeGreaterThanOrEqual(48 + 8);
  });

  it('clears the gallery cluster listing from its button', async () => {
    selectCluster('sunset beach');
    await renderMap(null);

    const clear = chip()!.querySelector<HTMLButtonElement>('button[aria-label="Clear cluster selection"]')!;
    clear.focus();
    await act(() => clear.click());

    expect(workbench.patchValues).toHaveBeenCalledWith('gallery', CLEARED);
    // The button is about to unmount with the chip; focus stays on the map, where Esc keeps working.
    expect(document.activeElement).toBe(host!.querySelector('[data-image-map-surface]'));
  });

  it('moves the Esc binding to the new project when the runtime changes without a remount', async () => {
    await renderMap(null);
    await renderMap(null, makeViewProps('project-b'));

    const escBindings = registered.hotkeys.filter((entry) => entry.defaultKeys.includes('esc'));
    // Registrations carry the project they were made for; a binding left on the
    // old project's runtime would never resolve in the new one.
    expect(escBindings.map((entry) => [entry.runtime, entry.disposed])).toEqual([
      ['project-a', true],
      ['project-b', false],
    ]);
    expect(
      registered.commands
        .filter((entry) => entry.id === escBindings[0]!.commandId)
        .map((entry) => [entry.runtime, entry.disposed])
    ).toEqual([
      ['project-a', true],
      ['project-b', false],
    ]);
  });

  it('shows nothing when the gallery is not listing a cluster', async () => {
    workbench.galleryValues = { searchTerm: 'cats' };
    await renderMap(null);

    expect(chip()).toBeNull();
  });

  it('binds Esc to the clear, which leaves an ordinary search alone', async () => {
    workbench.galleryValues = { searchTerm: 'cats' };
    await renderMap(null);

    const hotkey = registered.hotkeys.find((entry) => entry.defaultKeys.includes('esc'));
    const command = registered.commands.find((entry) => entry.id === hotkey?.commandId);
    expect(command).toBeDefined();

    // Esc with no cluster selected must not wipe the user's search.
    await act(() => {
      command!.handler();
    });
    expect(workbench.patchValues).not.toHaveBeenCalled();

    selectCluster('sunset beach');
    await act(() => {
      command!.handler();
    });
    expect(workbench.patchValues).toHaveBeenCalledWith('gallery', CLEARED);
  });
});
