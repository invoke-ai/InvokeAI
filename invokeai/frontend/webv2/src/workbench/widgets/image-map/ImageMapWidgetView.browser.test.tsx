import type { ImageMapState } from '@workbench/image-map/api';
import type * as ImageMapStoreModule from '@workbench/image-map/imageMapStore';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

const workbench = vi.hoisted(() => ({
  galleryValues: {} as Record<string, unknown>,
  projectId: 'project-a',
  patchValues: vi.fn(),
  updateGallerySettings: vi.fn(),
}));

const recovery = vi.hoisted(() => ({ status: vi.fn(), points: vi.fn() }));
vi.mock('@workbench/image-map/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  fetchImageMapStatus: recovery.status,
  fetchImageMapPoints: recovery.points,
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => workbench.projectId,
  useWidgetValuesSelector: (widgetId: string, select: (values: Record<string, unknown>) => unknown) =>
    select(widgetId === 'gallery' ? workbench.galleryValues : {}),
  useWorkbenchCommands: () => ({
    gallery: { updateSettings: workbench.updateGallerySettings },
    widgets: { patchValues: workbench.patchValues },
  }),
  useWorkbenchQueries: () => ({ getSnapshot: () => ({ activeProject: {} }) }),
}));
vi.mock('@workbench/widgetState', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProjectWidgetValues: () => workbench.galleryValues,
}));

vi.mock('./ImageMapPlot', () => ({ default: () => <div data-testid="plot" /> }));

const models = vi.hoisted(() => {
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
    canManageAppConfig: true,
    canManageModels: true,
    ensureStartersLoaded: vi.fn(),
    openAddModelsSearch: vi.fn(),
  };
});

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useCapabilities: () => ({ canManageAppConfig: models.canManageAppConfig, canManageModels: models.canManageModels }),
}));

vi.mock('@features/models', async (importOriginal) => {
  const { useSyncExternalStore } = await import('react');

  return {
    ...(await importOriginal<Record<string, unknown>>()),
    ensureStartersLoaded: models.ensureStartersLoaded,
    useActiveInstallSources: () =>
      useSyncExternalStore(models.activeSources.subscribe, models.activeSources.get, models.activeSources.get),
    useOpenAddModelsSearch: () => (models.canManageModels ? models.openAddModelsSearch : null),
    useStartersSelector: (selector: (snapshot: unknown) => unknown) =>
      selector({ response: { starter_models: STARTERS } }),
  };
});

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
    requestImageMapRecompute: vi.fn(),
    setClusterLabelsEnabled: vi.fn(),
    setImageMapMediaKind: vi.fn(),
  };
});

import { registerImageCluster } from '@features/gallery/contracts';
import { imageMapStore, refreshImageIndexStatus, requestImageMapRecompute } from '@workbench/image-map/imageMapStore';

import { ImageMapWidgetView } from './ImageMapWidgetView';

await i18next.use(initReactI18next).init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const registered = {
  commands: [] as { handler: () => unknown; id: string; runtime: string; disposed: boolean }[],
  hotkeys: [] as { commandId: string; defaultKeys: string[]; runtime: string; disposed: boolean }[],
};
const makeViewProps = (runtimeName: string, region = 'center') =>
  ({
    runtime: {
      instanceId: 'image-map',
      region,
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
  state: Extract<ImageMapState, 'computing' | 'disabled' | 'model_missing'>,
  modelName = 'clip-vit-large-patch14'
) => ({
  clusterEps: null,
  modelId: null,
  modelName: state === 'model_missing' ? modelName : null,
  pointCount: 0,
  points: [],
  stale: false,
  state,
  updatedAt: null,
  visibleHash: null,
});

const renderState = async (
  state: Extract<ImageMapState, 'computing' | 'disabled' | 'model_missing'>,
  modelName?: string
) => {
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

const buttonLabels = (): string[] =>
  Array.from(host?.querySelectorAll('button') ?? []).map((button) => button.textContent ?? '');

beforeEach(() => {
  vi.clearAllMocks();
  workbench.galleryValues = {};
  workbench.projectId = 'project-a';
  registered.commands = [];
  registered.hotkeys = [];
  models.activeSources.set([]);
  models.canManageAppConfig = true;
  models.canManageModels = true;
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

describe('Image Map computing state', () => {
  it('lets Check again bypass the refit cooldown a failed fit waits behind', async () => {
    await renderState('computing');

    await act(() => {
      Array.from(host?.querySelectorAll('button') ?? [])
        .find((button) => button.textContent === 'Check again')
        ?.click();
    });

    expect(requestImageMapRecompute).toHaveBeenCalledTimes(1);
    expect(refreshImageIndexStatus).toHaveBeenCalledTimes(1);
  });
});

describe('Image Map media filter', () => {
  it('says the shared media filter emptied the map and offers to show both kinds', async () => {
    workbench.galleryValues = { mediaFilter: 'video' };
    imageMapStore.setSnapshot({
      clusterLabels: null,
      clusterLabelsEps: null,
      clusterLabelsHash: null,
      data: { ...dataFor('disabled'), state: 'empty' },
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

    expect(host?.textContent).toContain('No videos to map');
    expect(host?.textContent).not.toContain('Nothing to map yet');

    const trigger = document.createElement('button');
    trigger.id = 'image-map-media-filter-trigger';
    document.body.append(trigger);

    await page.getByRole('button', { name: 'Show all media' }).click();

    expect(workbench.updateGallerySettings).toHaveBeenCalledExactlyOnceWith({ mediaFilter: 'all' });
    expect(document.activeElement).toBe(trigger);
    trigger.remove();
  });
});

describe('Image Map missing-model link', () => {
  it('replaces stalled indexing with the install link on a status poll and recovers after reinstall', async () => {
    const actual = await vi.importActual<typeof ImageMapStoreModule>('@workbench/image-map/imageMapStore');
    await renderState('disabled');
    await act(() => {
      imageMapStore.patchSnapshot({
        data: { ...dataFor('disabled'), state: 'computing' },
        indexCounts: { embedded: 4, failed: 0, pending: 6, total: 10 },
        indexUpdatedAt: Date.now() - 120_000,
      });
    });
    expect(host?.textContent).toContain('Indexing gallery');
    recovery.status.mockResolvedValue({
      index: null,
      modelId: null,
      modelName: 'DFN2B-CLIP-ViT-L-14-39B',
      state: 'model_missing',
    });
    await act(async () => {
      actual.refreshImageIndexStatus();
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
    expect(host?.textContent).toContain('Embedding model not installed');
    expect(host?.querySelector('[role="progressbar"]')).toBeNull();
    const link = host?.querySelector('button');
    expect(link?.textContent).toBe('DFN2B-CLIP-ViT-L-14-39B');
    await act(() => link?.click());
    expect(models.openAddModelsSearch).toHaveBeenCalledWith('DFN2B-CLIP-ViT-L-14-39B');
    await page.screenshot({ path: '../../../../artifacts/image-map/model-removed.png' });

    actual.setClusterLabelsEnabled(false);
    recovery.status.mockResolvedValue({ index: null, modelId: null, modelName: null, state: 'empty' });
    recovery.points.mockResolvedValue({ ...dataFor('disabled'), state: 'empty' });
    await act(async () => {
      actual.refreshImageIndexStatus();
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
    expect(host?.textContent).toContain('Nothing to map yet');
    expect(host?.textContent).not.toContain('Embedding model not installed');
    actual.setClusterLabelsEnabled(true);
  });

  it('opens Add Models on the configured encoder instead of downloading it', async () => {
    await renderState('model_missing', 'DFN2B-CLIP-ViT-L-14-39B');

    expect(host?.textContent).toContain(
      'To enable image indexing, install the image encoder model DFN2B-CLIP-ViT-L-14-39B from the Model Manager to build a semantic index of your gallery.'
    );

    const link = host?.querySelector('button');

    expect(link?.textContent).toBe('DFN2B-CLIP-ViT-L-14-39B');

    await act(() => link?.click());

    expect(models.openAddModelsSearch).toHaveBeenCalledWith('DFN2B-CLIP-ViT-L-14-39B');
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
        modelId: null,
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
    await renderMapWithCounts({ embedded: 1204, failed: 0, pending: 16846, total: 18050 });

    expect(host?.querySelector('[data-testid="plot"]')).not.toBeNull();
    expect(host?.textContent).toContain('indexing 1,204/18,050');
    expect(host?.textContent).not.toContain('Indexing gallery');
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
        modelId: null,
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
    expect(getComputedStyle(chip()!.querySelector('[aria-hidden="true"]')!).backgroundColor).toBe('rgb(225, 87, 89)');
    const badge = host?.querySelector('[role="progressbar"]');
    expect(chip()!.compareDocumentPosition(badge!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(badge!.getBoundingClientRect().top).toBeGreaterThanOrEqual(chip()!.getBoundingClientRect().bottom);
  });

  it('clears the gallery cluster listing from its button', async () => {
    selectCluster('sunset beach');
    await renderMap(null);

    const clear = chip()!.querySelector<HTMLButtonElement>('button[aria-label="Clear cluster selection"]')!;
    clear.focus();
    await act(() => clear.click());

    expect(workbench.patchValues).toHaveBeenCalledWith('gallery', CLEARED);
    expect(document.activeElement).toBe(host!.querySelector('[data-image-map-surface]'));
  });

  it('binds Esc only while a cluster is selected, so with none it reaches Back to stage', async () => {
    const escBound = () =>
      registered.hotkeys.filter((entry) => entry.defaultKeys.includes('esc') && !entry.disposed).length;

    await renderMap(null);
    expect(escBound()).toBe(0);

    selectCluster('sunset beach');
    await renderMap(null);
    expect(escBound()).toBe(1);
    registered.commands.find((entry) => !entry.disposed)!.handler();
    expect(workbench.patchValues).toHaveBeenCalledWith('gallery', CLEARED);

    workbench.galleryValues = {};
    await renderMap(null);
    expect(escBound()).toBe(0);
  });

  it.each([
    ['project', () => (workbench.projectId = 'project-b'), 'center'],
    ['region', () => undefined, 'right'],
  ])(
    'moves the Esc binding to the new runtime when its %s changes without a remount',
    async (_what, change, region) => {
      selectCluster('sunset beach');
      await renderMap(null);
      change();
      await renderMap(null, makeViewProps('runtime-b', region));

      const escBindings = registered.hotkeys.filter((entry) => entry.defaultKeys.includes('esc'));
      expect(escBindings.map((entry) => [entry.runtime, entry.disposed])).toEqual([
        ['project-a', true],
        ['runtime-b', false],
      ]);
      expect(
        registered.commands
          .filter((entry) => entry.id === escBindings[0]!.commandId)
          .map((entry) => [entry.runtime, entry.disposed])
      ).toEqual([
        ['project-a', true],
        ['runtime-b', false],
      ]);
    }
  );
});
