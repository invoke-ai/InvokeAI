/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GalleryImage, GalleryImageItem, GalleryVideoItem } from '@features/gallery';
import type * as GalleryModule from '@features/gallery';
import type * as GalleryBackendModule from '@features/gallery/data/backend';
import type * as GalleryQueriesModule from '@features/gallery/queries';
import type * as IdentityModule from '@features/identity';
import type * as ClipboardModule from '@platform/browser/clipboard';
import type { ImageActions, ImageRecallCapabilities } from '@workbench/image-actions';

import { ChakraProvider } from '@chakra-ui/react';
import { galleryKeys } from '@features/gallery/queries';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { EMPTY_IMAGE_RECALL_CAPABILITIES } from '@workbench/image-actions';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PreviewDetailsPanel } from './PreviewDetailsPanel';

const galleryMocks = vi.hoisted(() => ({
  imageMetadata: vi.fn(),
  imageWorkflow: vi.fn(),
  videoMetadata: vi.fn(),
  videoWorkflow: vi.fn(),
}));
const identityMocks = vi.hoisted(() => ({ accountEpoch: 7 }));
const clipboardMocks = vi.hoisted(() => ({ copyText: vi.fn<(text: string) => Promise<void>>() }));

vi.mock('@platform/browser/clipboard', async (importOriginal) => ({
  ...(await importOriginal<typeof ClipboardModule>()),
  canCopyText: () => true,
  copyText: clipboardMocks.copyText,
}));

vi.mock('@features/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof GalleryModule>();

  return {
    ...actual,
    galleryImages: { ...actual.galleryImages, workflow: galleryMocks.imageWorkflow },
    galleryVideos: { ...actual.galleryVideos, workflow: galleryMocks.videoWorkflow },
  };
});
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackendModule>()),
  getGalleryImageMetadata: galleryMocks.imageMetadata,
  getGalleryImageWorkflow: galleryMocks.imageWorkflow,
  getGalleryVideoMetadata: galleryMocks.videoMetadata,
}));
vi.mock('@features/identity', async (importOriginal) => {
  const actual = await importOriginal<typeof IdentityModule>();

  return {
    ...actual,
    useAuthSession: () => ({
      accountEpoch: identityMocks.accountEpoch,
      multiuserEnabled: true,
      phase: 'ready',
      sessionExpired: false,
      setupRequired: false,
      strictPasswordChecking: true,
      user: null,
    }),
  };
});
vi.mock('@features/gallery/queries', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryQueriesModule>()),
  imageIndexAvailabilityOptions: () => ({
    queryFn: () => ({ state: 'disabled' }),
    queryKey: ['test-image-index-availability'],
  }),
}));

const imageItem: GalleryImageItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-07-30T12:00:00Z',
  fullUrl: '/images/still.png',
  height: 768,
  isIntermediate: false,
  kind: 'image',
  name: 'still.png',
  sourceQueueItemId: 'queue-image',
  starred: false,
  thumbnailUrl: '/thumbnails/still.webp',
  width: 512,
};
const actionImage: GalleryImage = {
  boardId: imageItem.boardId,
  height: imageItem.height,
  imageCategory: imageItem.category,
  imageName: imageItem.name,
  imageUrl: imageItem.fullUrl,
  queuedAt: imageItem.createdAt,
  sourceQueueItemId: imageItem.sourceQueueItemId ?? 'backend-gallery',
  starred: imageItem.starred,
  thumbnailUrl: imageItem.thumbnailUrl,
  width: imageItem.width,
};
const videoItem: GalleryVideoItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-07-30T11:00:00Z',
  durationSeconds: 15,
  fullUrl: '/videos/clip.mp4',
  height: 1080,
  isIntermediate: false,
  kind: 'video',
  name: 'clip.mp4',
  starred: false,
  thumbnailUrl: '/thumbnails/clip.webp',
  width: 1920,
};
const ALL_RECALL_CAPABILITIES: ImageRecallCapabilities = {
  all: true,
  clipSkip: true,
  dimensions: true,
  prompts: true,
  remix: true,
  seed: true,
  workflow: true,
};

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      metadata: { widgets: { preview: { metadataLabels: { generation_mode: 'Generation path' } } } },
      translation: {
        common: { countOfTotal: '{{count}} of {{total}}', copy: 'Copy', retry: 'Retry' },
        widgets: {
          gallery: {
            itemActions: {
              recall: {
                verbs: {
                  all: 'Recall all',
                  clipSkip: 'Use CLIP skip',
                  dimensions: 'Use size',
                  prompts: 'Use prompt',
                  positivePrompt: 'Use positive prompt',
                  negativePrompt: 'Use negative prompt',
                  remix: 'Remix image',
                  seed: 'Use seed',
                },
              },
            },
          },
          preview: {
            details: 'Details',
            fields: {
              fps: 'Frame rate',
              frames: 'Frame count',
              model: 'Checkpoint',
              prompt: 'Prompt text',
              seed: 'Seed value',
              sourceRun: 'Queue run',
            },
            framesPerSecond: '{{count}} fps',
            graph: 'Graph',
            graphJsonLabel: 'Video graph JSON',
            itemCount_one: '{{count}} item',
            itemCount_other: '{{count}} items',
            loadingMetadata: 'Loading metadata',
            metadata: 'Metadata',
            metadataFailed: 'Could not load the metadata',
            metadataJsonLabel: 'Video metadata JSON',
            nextItemInBoard: 'Next item in board',
            previousItemInBoard: 'Previous item in board',
            recallNotAvailable: 'Not recallable for this image',
            retryMetadata: 'Retry loading the metadata',
            retryWorkflow: 'Retry loading the workflow and graph',
            videoDuration: 'Duration {{duration}}',
            workflow: 'Workflow',
            workflowFailed: 'Could not load the workflow and graph',
            workflowJsonLabel: 'Video workflow JSON',
          },
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let stage: HTMLDivElement | null = null;
let queryClient: QueryClient;
let actions: ImageActions;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void, delay = 100): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, delay);
    });
  });

const POSITION = { boardItemCount: 2, isLoadingBoard: false, selectedIndex: 0 };

const renderDetails = async ({
  isOpen,
  item,
  localization = i18n,
}: {
  isOpen: boolean;
  item: GalleryImageItem | GalleryVideoItem;
  localization?: typeof i18n;
}): Promise<void> => {
  await interact(() => {
    root?.render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={localization}>
          <ChakraProvider value={system}>
            <PreviewDetailsPanel
              actions={actions}
              image={item.kind === 'image' ? actionImage : null}
              isOpen={isOpen}
              item={item}
              position={POSITION}
              stageElement={stage}
              onOpenChange={() => undefined}
            />
          </ChakraProvider>
        </I18nextProvider>
      </QueryClientProvider>
    );
  });
};

beforeEach(() => {
  identityMocks.accountEpoch = 7;
  galleryMocks.imageMetadata.mockReset().mockResolvedValue(null);
  galleryMocks.imageWorkflow.mockReset().mockResolvedValue({ graph: null, workflow: null });
  galleryMocks.videoMetadata.mockReset().mockResolvedValue(null);
  galleryMocks.videoWorkflow.mockReset().mockResolvedValue({ graph: null, workflow: null });
  clipboardMocks.copyText.mockReset().mockResolvedValue(undefined);
  actions = {
    deriveImageRecallCapabilities: vi.fn(() => ALL_RECALL_CAPABILITIES),
    recallImageData: vi.fn(() => Promise.resolve()),
  } as unknown as ImageActions;
  queryClient = new QueryClient({
    defaultOptions: { queries: { gcTime: Infinity, retry: false } },
  });
  host = document.createElement('div');
  stage = document.createElement('div');
  Object.assign(stage.style, { height: '480px', position: 'relative', width: '640px' });
  document.body.append(host, stage);
  root = createRoot(host);
});

afterEach(async () => {
  await interact(() => root?.unmount(), 0);
  queryClient.clear();
  host?.remove();
  stage?.remove();
  host = null;
  root = null;
  stage = null;
});

describe('Preview query-driven Details', () => {
  const sourceGraph = {
    id: 'source',
    nodes: {
      iterator: { id: 'iterator', type: 'iterate', collection: [17, 18] },
      math: { id: 'math', type: 'add', a: 0, b: 1 },
      noise: { id: 'noise', type: 'noise', seed: 0 },
      collector: { id: 'collector', type: 'collect' },
    },
    edges: [
      { source: { node_id: 'iterator', field: 'item' }, destination: { node_id: 'math', field: 'a' } },
      { source: { node_id: 'math', field: 'value' }, destination: { node_id: 'noise', field: 'seed' } },
      { source: { node_id: 'noise', field: 'noise' }, destination: { node_id: 'collector', field: 'item' } },
    ],
  };

  it.each([
    {
      label: 'Executed controls',
      expected: 'Depth model',
      metadata: { control_layers: [{ model: { name: 'Depth model' }, type: 'controlnet', weight: 0.75 }] },
    },
    {
      label: 'IP-Adapters',
      expected: 'Portrait model',
      metadata: {
        ipAdapters: [
          {
            ip_adapter_model: { name: 'Portrait model' },
            image: { image_name: 'portrait.png' },
            clip_vision_model: 'ViT-G',
            method: 'style_precise',
            weight: 0.75,
            begin_step_percent: 0.2,
            end_step_percent: 0.9,
          },
        ],
      },
    },
    ...(['array', 'entities'] as const).map((shape) => {
      const reference = {
        id: 'global-style',
        type: 'reference_image',
        name: 'Global style',
        ipAdapter: {
          type: 'ip_adapter',
          model: { name: 'Historical portrait' },
          image: { image_name: 'portrait.png', width: 320, height: 200 },
          clipVisionModel: 'ViT-G',
          method: 'style_precise',
          weight: 0.75,
          beginEndStepPct: [0.2, 0.9],
        },
      };
      return {
        label: 'Reference images',
        expected: 'Historical portrait',
        metadata: {
          canvas_v2_metadata: {
            rasterLayers: [],
            controlLayers: [],
            inpaintMasks: [],
            regionalGuidance: [],
            referenceImages: shape === 'array' ? [reference] : { entities: [reference] },
          },
        },
      };
    }),
  ])(
    'shows readable $label metadata from the emitted English namespace beside root actions',
    async ({ label, expected, metadata }) => {
      const localization = createInstance().use(initReactI18next);
      await localization.init({
        initAsync: false,
        lng: 'en',
        resources: {
          en: {
            translation: await fetch('/locales/en.json').then((response) => response.json()),
            metadata: await fetch('/locales/en.metadata.json').then((response) => response.json()),
          },
        },
      });
      galleryMocks.imageMetadata.mockResolvedValueOnce(metadata);
      await renderDetails({ isOpen: true, item: imageItem, localization });
      await act(async () => {
        await vi.waitFor(() => expect(document.body.textContent).toContain(expected));
      });
      const controlRow = [...document.querySelectorAll('.chakra-data-list__item')].find((row) =>
        row.textContent?.includes(expected)
      );
      expect(controlRow?.querySelector('dt')?.textContent).toBe(label);
      if (label !== 'Executed controls') {
        expect(controlRow?.textContent).toContain('portrait.png · ViT-G · style_precise · Weight 0.75 · Steps 0.2–0.9');
      }
      expect(controlRow?.textContent).toContain('Weight 0.75');
      expect(controlRow?.querySelector('button[aria-label="Copy"]')).not.toBeNull();
      if (label === 'Reference images') {
        expect(document.body.textContent).toContain('0 layers');
        expect(document.body.textContent).not.toContain('Global style');
      }
    }
  );

  it.each([{}, { resolved_inputs: null }])('shows the source graph for legacy image payloads: %j', async (legacy) => {
    galleryMocks.imageWorkflow.mockResolvedValueOnce({ graph: JSON.stringify(sourceGraph), workflow: null, ...legacy });

    await renderDetails({ isOpen: true, item: imageItem });
    await act(async () => {
      await vi.waitFor(() => expect(getButton('Graph').disabled).toBe(false));
    });
    await interact(() => getButton('Graph').click());

    const graphRegion = document.querySelector('[role="region"][aria-label="Video graph JSON"]');
    expect(graphRegion).not.toBeNull();
    expect(JSON.parse(graphRegion!.textContent!)).toEqual(sourceGraph);
  });

  it('aborts the sole image metadata transport on close without starting duplicate capability work', async () => {
    const metadata = deferred<null>();
    galleryMocks.imageMetadata.mockReturnValueOnce(metadata.promise);

    await renderDetails({ isOpen: true, item: imageItem });

    const metadataSignal = galleryMocks.imageMetadata.mock.calls[0]?.[1] as AbortSignal;
    expect(galleryMocks.imageMetadata).toHaveBeenCalledOnce();

    await renderDetails({ isOpen: false, item: imageItem });

    expect(metadataSignal.aborted).toBe(true);
    await interact(() => {
      metadata.resolve(null);
    }, 0);
  });

  it('aborts and isolates stale item results behind qualified keys', async () => {
    const oldMetadata = deferred<Record<string, unknown> | null>();
    const oldWorkflow = deferred<{ graph: string | null; workflow: string | null }>();
    galleryMocks.videoMetadata.mockImplementation((name: string) =>
      name === 'clip.mp4' ? oldMetadata.promise : Promise.resolve({ marker: 'new-item' })
    );
    galleryMocks.videoWorkflow.mockImplementation((name: string) =>
      name === 'clip.mp4'
        ? oldWorkflow.promise
        : Promise.resolve({ graph: '{"marker":"new-graph"}', workflow: '{"marker":"new-workflow"}' })
    );
    await renderDetails({ isOpen: true, item: videoItem });
    const oldSignal = galleryMocks.videoMetadata.mock.calls[0]?.[1] as AbortSignal;
    const nextItem = {
      ...videoItem,
      fullUrl: '/videos/next.mp4',
      name: 'next.mp4',
      thumbnailUrl: '/thumbnails/next.webp',
    };

    await renderDetails({ isOpen: true, item: nextItem });

    expect(oldSignal.aborted).toBe(true);
    expect(galleryMocks.videoMetadata).toHaveBeenNthCalledWith(2, 'next.mp4', expect.any(AbortSignal));
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('new-item'));
    });

    await interact(() => {
      oldMetadata.resolve({ marker: 'stale-item' });
      oldWorkflow.resolve({ graph: '{"marker":"stale-graph"}', workflow: '{"marker":"stale-workflow"}' });
    });
    expect(document.body.textContent).not.toContain('stale-item');
  });

  it('aborts and isolates stale account results behind the account epoch', async () => {
    const oldMetadata = deferred<Record<string, unknown> | null>();
    const oldWorkflow = deferred<{ graph: string | null; workflow: string | null }>();
    galleryMocks.videoMetadata
      .mockReturnValueOnce(oldMetadata.promise)
      .mockResolvedValueOnce({ marker: 'new-account' });
    galleryMocks.videoWorkflow
      .mockReturnValueOnce(oldWorkflow.promise)
      .mockResolvedValueOnce({ graph: '{"epoch":8}', workflow: '{"epoch":8}' });
    await renderDetails({ isOpen: true, item: videoItem });
    const oldSignal = galleryMocks.videoMetadata.mock.calls[0]?.[1] as AbortSignal;

    identityMocks.accountEpoch = 8;
    await renderDetails({ isOpen: true, item: videoItem });

    expect(oldSignal.aborted).toBe(true);
    expect(galleryMocks.videoMetadata).toHaveBeenCalledTimes(2);
    expect(
      queryClient.getQueryCache().find({ queryKey: ['preview', 'details', 8, 'video:clip.mp4', 'workflow'] })
    ).toBeDefined();
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('new-account'));
    });

    await interact(() => {
      oldMetadata.resolve({ marker: 'stale-account' });
      oldWorkflow.resolve({ graph: '{"epoch":7}', workflow: '{"epoch":7}' });
    });
    expect(document.body.textContent).not.toContain('stale-account');
  });

  it('keeps parsed image rows, source run, and recall controls while moving the fetch to Query', async () => {
    galleryMocks.imageMetadata.mockResolvedValueOnce({
      model: { name: 'Test Model' },
      positive_prompt: 'query-driven prompt',
      negative_prompt: 'recalled negative prompt',
      seed: 42,
    });

    await renderDetails({ isOpen: true, item: imageItem });

    expect(galleryMocks.imageMetadata).toHaveBeenCalledWith('still.png', expect.any(AbortSignal));
    expect(
      queryClient.getQueryCache().find({ queryKey: galleryKeys.imageMetadata(captureAccountScope(), 'still.png') })
    ).toBeDefined();
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('query-driven prompt'));
    });
    expect(document.body.textContent).toContain('Test Model');
    expect(document.body.textContent).toContain('Queue run');
    expect(document.body.textContent).toContain('queue-image');

    await interact(() => document.querySelector<HTMLButtonElement>('[aria-label="Use positive prompt"]')?.click());
    expect(actions.recallImageData).toHaveBeenLastCalledWith(actionImage, 'positivePrompt');
    await interact(() => document.querySelector<HTMLButtonElement>('[aria-label="Use negative prompt"]')?.click());
    expect(actions.recallImageData).toHaveBeenLastCalledWith(actionImage, 'negativePrompt');
  });

  it('keeps prompt rows readable without offering recall when no Generate target is available', async () => {
    galleryMocks.imageMetadata.mockResolvedValueOnce({
      positive_prompt: 'readable positive',
      negative_prompt: 'readable negative',
    });
    vi.mocked(actions.deriveImageRecallCapabilities).mockReturnValue(EMPTY_IMAGE_RECALL_CAPABILITIES);

    await renderDetails({ isOpen: true, item: imageItem });
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('readable positive'));
    });

    expect(document.body.textContent).toContain('readable negative');
    expect(document.querySelector('[aria-label="Use positive prompt"]')).toBeNull();
    expect(document.querySelector('[aria-label="Use negative prompt"]')).toBeNull();
  });

  it('keeps the video workflow and graph tabs when its metadata cannot be read, and retries the metadata', async () => {
    galleryMocks.videoMetadata
      .mockRejectedValueOnce(new Error('metadata unavailable'))
      .mockResolvedValueOnce({ codec: 'h264' });
    galleryMocks.videoWorkflow.mockResolvedValueOnce({ graph: '{"nodes":[]}', workflow: '{"name":"clip"}' });

    await renderDetails({ isOpen: true, item: videoItem });
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('Could not load the metadata'));
    });

    expect(getButton('Workflow').getAttribute('aria-disabled')).not.toBe('true');
    expect(getButton('Graph').getAttribute('aria-disabled')).not.toBe('true');
    expect(getButton('Metadata').getAttribute('aria-disabled')).toBe('true');
    expect(document.body.textContent).not.toContain('Could not load the workflow and graph');

    const retry = document.querySelector<HTMLButtonElement>('button[aria-label="Retry loading the metadata"]');
    expect(retry).not.toBeNull();
    retry?.focus();
    await interact(() => retry?.click());

    await act(async () => {
      await vi.waitFor(() => expect(getButton('Metadata').getAttribute('aria-disabled')).not.toBe('true'));
    });
    expect(galleryMocks.videoMetadata).toHaveBeenCalledTimes(2);
    expect(galleryMocks.videoWorkflow).toHaveBeenCalledOnce();
    expect(document.body.textContent).not.toContain('Could not load the metadata');
    expect(document.body.textContent).toContain('"codec": "h264"');
    await vi.waitFor(() => expect(document.activeElement).toBe(getButton('Metadata')));
  });

  it('recalls a single field from its metadata row', async () => {
    galleryMocks.imageMetadata.mockResolvedValueOnce({
      model: { name: 'Test Model' },
      positive_prompt: 'row prompt',
      seed: 42,
    });

    await renderDetails({ isOpen: true, item: imageItem });
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('row prompt'));
    });

    const seedRecall = document.querySelector<HTMLButtonElement>('[aria-label="Use seed"]');
    expect(seedRecall).not.toBeNull();
    expect(document.querySelector('[aria-label="Use positive prompt"]')).not.toBeNull();
    expect(document.querySelector('[aria-label="Use negative prompt"]')).toBeNull();
    expect(document.querySelector('[aria-label="Use size"]')).toBeNull();

    const modelRow = [...(document.querySelectorAll('.chakra-data-list__item') ?? [])].find((row) =>
      row.textContent?.includes('Test Model')
    );
    expect(modelRow?.querySelectorAll('button')).toHaveLength(1);
    expect(modelRow?.querySelector('[aria-label="Copy"]')).not.toBeNull();

    await interact(() => seedRecall?.click());
    expect(actions.recallImageData).toHaveBeenCalledWith(actionImage, 'seed');
  });
});

const getButton = (text: string): HTMLButtonElement => {
  const button = Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find(
    (candidate) => candidate.textContent?.trim() === text
  );

  if (!button) {
    throw new Error(`Expected button: ${text}`);
  }

  return button;
};

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};
