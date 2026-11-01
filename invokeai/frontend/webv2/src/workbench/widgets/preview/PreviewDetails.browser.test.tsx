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
// The transport under the gallery's shared metadata queries.
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackendModule>()),
  getGalleryImageMetadata: galleryMocks.imageMetadata,
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
const NO_RECALL_CAPABILITIES: ImageRecallCapabilities = {
  all: false,
  clipSkip: false,
  dimensions: false,
  prompts: false,
  remix: false,
  seed: false,
  workflow: false,
};

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
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
                  remix: 'Remix image',
                  seed: 'Use seed',
                },
              },
            },
          },
          preview: {
            details: 'Details',
            // Distinct from the English catalog, so the rows provably read their labels from i18n.
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
/** The Preview stage Details pins itself into. */
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
}: {
  isOpen: boolean;
  item: GalleryImageItem | GalleryVideoItem;
}): Promise<void> => {
  await interact(() => {
    root?.render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
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

  it("shows this image's resolved Noise and math inputs in the Graph tab", async () => {
    galleryMocks.imageWorkflow.mockResolvedValueOnce({
      graph: JSON.stringify(sourceGraph),
      resolved_inputs: JSON.stringify({
        id: 'resolved',
        nodes: {
          math: { id: 'math', type: 'add', a: 17, b: 1 },
          noise: { id: 'noise', type: 'noise', seed: 18 },
        },
        edges: [],
      }),
      workflow: null,
    });

    await renderDetails({ isOpen: true, item: imageItem });
    await act(async () => {
      await vi.waitFor(() => expect(getButton('Graph').disabled).toBe(false));
    });
    await interact(() => getButton('Graph').click());

    const graphRegion = document.querySelector('[role="region"][aria-label="Video graph JSON"]');
    expect(graphRegion).not.toBeNull();
    expect(JSON.parse(graphRegion!.textContent!)).toEqual({
      id: 'resolved',
      nodes: {
        math: { id: 'math', type: 'add', a: 17, b: 1 },
        noise: { id: 'noise', type: 'noise', seed: 18 },
      },
      edges: [],
    });
  });

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

  it('shows the video Details disclosure but starts no request while it is closed', async () => {
    await renderDetails({ isOpen: false, item: videoItem });

    expect(galleryMocks.videoMetadata).not.toHaveBeenCalled();
    expect(galleryMocks.videoWorkflow).not.toHaveBeenCalled();
    expect(document.querySelector('button[aria-label="Details"]')).not.toBeNull();
  });

  it('keeps the disabled image recall skeleton mounted while Details is pending', async () => {
    galleryMocks.imageMetadata.mockReturnValueOnce(
      new Promise(() => {
        // Intentionally pending.
      })
    );

    await renderDetails({ isOpen: true, item: imageItem });

    expect(document.body.textContent).toContain('Loading metadata');
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

  it('starts video metadata and workflow/graph concurrently, metadata under the shared gallery entry', async () => {
    const metadata = deferred<Record<string, unknown> | null>();
    const workflow = deferred<{ graph: string | null; workflow: string | null }>();
    galleryMocks.videoMetadata.mockReturnValueOnce(metadata.promise);
    galleryMocks.videoWorkflow.mockReturnValueOnce(workflow.promise);

    await renderDetails({ isOpen: true, item: videoItem });

    expect(galleryMocks.videoMetadata).toHaveBeenCalledOnce();
    expect(galleryMocks.videoWorkflow).toHaveBeenCalledOnce();
    expect(galleryMocks.videoMetadata.mock.calls[0]?.[0]).toBe('clip.mp4');
    expect(galleryMocks.videoWorkflow.mock.calls[0]?.[0]).toBe('clip.mp4');
    expect(galleryMocks.videoMetadata.mock.calls[0]?.[1]).toBeInstanceOf(AbortSignal);
    expect(galleryMocks.videoWorkflow.mock.calls[0]?.[1]).toBeInstanceOf(AbortSignal);
    // The gallery's shared entry, which the video's menu and recall commands read too.
    expect(
      queryClient.getQueryCache().find({ queryKey: galleryKeys.videoMetadata(captureAccountScope(), 'clip.mp4') })
    ).toBeDefined();
    expect(
      queryClient.getQueryCache().find({ queryKey: ['preview', 'details', 7, 'video:clip.mp4', 'workflow'] })
    ).toBeDefined();

    await interact(() => {
      metadata.resolve({ codec: 'h264' });
      workflow.resolve({ graph: '{"nodes":[]}', workflow: '{"name":"clip"}' });
    });
  });

  it('aborts in-flight supported transports when Details closes', async () => {
    const metadata = deferred<Record<string, unknown> | null>();
    const workflow = deferred<{ graph: string | null; workflow: string | null }>();
    galleryMocks.videoMetadata.mockReturnValueOnce(metadata.promise);
    galleryMocks.videoWorkflow.mockReturnValueOnce(workflow.promise);
    await renderDetails({ isOpen: true, item: videoItem });
    const metadataSignal = galleryMocks.videoMetadata.mock.calls[0]?.[1] as AbortSignal;
    const workflowSignal = galleryMocks.videoWorkflow.mock.calls[0]?.[1] as AbortSignal;

    await renderDetails({ isOpen: false, item: videoItem });

    expect(metadataSignal.aborted).toBe(true);
    expect(workflowSignal.aborted).toBe(true);
    expect(galleryMocks.videoMetadata).toHaveBeenCalledOnce();
    expect(galleryMocks.videoWorkflow).toHaveBeenCalledOnce();
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

  it('renders raw Metadata, Workflow, and Graph JSON tabs without image recall controls', async () => {
    galleryMocks.videoMetadata.mockResolvedValueOnce({ codec: 'h264', duration: 15 });
    galleryMocks.videoWorkflow.mockResolvedValueOnce({
      graph: '{"nodes":[{"id":"video"}]}',
      workflow: '{"name":"Video workflow"}',
    });

    await renderDetails({ isOpen: true, item: videoItem });

    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('"codec": "h264"'));
    });
    expect(getButton('Metadata')).not.toBeNull();
    expect(getButton('Workflow')).not.toBeNull();
    expect(getButton('Graph')).not.toBeNull();
    // A record with no parsed rows offers no Details tab.
    expect([...document.querySelectorAll('[role="tab"]')].map((tab) => tab.textContent)).toEqual([
      'Metadata',
      'Workflow',
      'Graph',
    ]);

    await interact(() => getButton('Workflow').click());
    expect(document.body.textContent).toContain('"name":"Video workflow"');
    await interact(() => getButton('Graph').click());
    expect(document.body.textContent).toContain('"id":"video"');
    expect(document.querySelector('[aria-label="Video graph JSON"]')).not.toBeNull();
  });

  it('opens a video on its parsed rows, which copy but offer no recall', async () => {
    galleryMocks.videoMetadata.mockResolvedValueOnce({
      fps: 16,
      generation_mode: 'wan_t2v',
      model: { base_model: 'wan', model_name: 'Wan 2.2 T2V' },
      num_frames: 81,
      positive_prompt: 'a red fox',
      seed: 7,
    });

    await renderDetails({ isOpen: true, item: videoItem });
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('a red fox'));
    });

    expect([...document.querySelectorAll('[role="tab"]')].map((tab) => tab.textContent)).toEqual([
      'Details',
      'Metadata',
      'Workflow',
      'Graph',
    ]);
    const rows: Record<string, Element> = Object.fromEntries(
      [...document.querySelectorAll('.chakra-data-list__item')].map((row) => [
        row.querySelector('dt')?.textContent,
        row,
      ])
    );
    expect(Object.keys(rows)).toEqual(['Prompt text', 'Checkpoint', 'Seed value', 'Frame count', 'Frame rate']);
    expect(rows['Checkpoint']?.textContent).toContain('Wan 2.2 T2V');
    expect(rows['Frame count']?.textContent).toContain('81');
    // Copy is each row's only action; video recall lives in its menu.
    expect(document.querySelector('[aria-label="Use prompt"]')).toBeNull();
    expect(rows['Frame rate']?.querySelectorAll('button')).toHaveLength(1);

    await interact(() => rows['Frame rate']?.querySelector<HTMLButtonElement>('[aria-label="Copy"]')?.click());
    expect(clipboardMocks.copyText).toHaveBeenCalledWith('16');
  });

  it('keeps parsed image rows, source run, and recall controls while moving the fetch to Query', async () => {
    galleryMocks.imageMetadata.mockResolvedValueOnce({
      model: { name: 'Test Model' },
      positive_prompt: 'query-driven prompt',
      seed: 42,
    });

    await renderDetails({ isOpen: true, item: imageItem });

    expect(galleryMocks.imageMetadata).toHaveBeenCalledWith('still.png', expect.any(AbortSignal));
    // The gallery's shared entry, which the image's menus and recall commands read too.
    expect(
      queryClient.getQueryCache().find({ queryKey: galleryKeys.imageMetadata(captureAccountScope(), 'still.png') })
    ).toBeDefined();
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('query-driven prompt'));
    });
    expect(document.body.textContent).toContain('Test Model');
    expect(document.body.textContent).toContain('Queue run');
    expect(document.body.textContent).toContain('queue-image');

    await interact(() => document.querySelector<HTMLButtonElement>('[aria-label="Use prompt"]')?.click());
    expect(actions.recallImageData).toHaveBeenCalledWith(actionImage, 'prompts');
  });

  it('keeps the image workflow and graph tabs when its metadata cannot be read', async () => {
    galleryMocks.imageMetadata.mockRejectedValueOnce(new Error('metadata unavailable'));
    galleryMocks.imageWorkflow.mockResolvedValueOnce({ graph: '{"nodes":[]}', workflow: '{"name":"still"}' });

    await renderDetails({ isOpen: true, item: imageItem });
    await act(async () => {
      await vi.waitFor(() => expect(getButton('Workflow').getAttribute('aria-disabled')).not.toBe('true'));
    });

    expect(getButton('Graph').getAttribute('aria-disabled')).not.toBe('true');
    expect(getButton('Metadata').getAttribute('aria-disabled')).toBe('true');
    // Recall still derives from what the image itself carries.
    expect(actions.deriveImageRecallCapabilities).toHaveBeenLastCalledWith(actionImage, null);
    expect(document.body.textContent).toContain('queue-image');
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
    // Nothing was chosen yet, so the panel moves to the tab the retry filled; the removed button hands focus there.
    expect(document.body.textContent).toContain('"codec": "h264"');
    await vi.waitFor(() => expect(document.activeElement).toBe(getButton('Metadata')));
  });

  it('offers a retry for each failed video read over no pane, and fills the tab a retry reads', async () => {
    galleryMocks.videoMetadata.mockRejectedValueOnce(new Error('metadata unavailable'));
    galleryMocks.videoWorkflow
      .mockRejectedValueOnce(new Error('workflow unavailable'))
      .mockResolvedValueOnce({ graph: '{"nodes":[]}', workflow: '{"name":"clip"}' });

    await renderDetails({ isOpen: true, item: videoItem });
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('Could not load the workflow and graph'));
    });

    expect(document.body.textContent).toContain('Could not load the metadata');
    const retries = [...document.querySelectorAll<HTMLButtonElement>('[role="alert"] button')].map((button) =>
      button.getAttribute('aria-label')
    );
    expect(retries).toEqual(['Retry loading the metadata', 'Retry loading the workflow and graph']);
    // Every tab is disabled; Metadata stays selected with no empty JSON pane behind it.
    expect(document.querySelector('[role="tab"][aria-selected="true"]')?.textContent).toBe('Metadata');
    expect(document.querySelector('[role="tabpanel"]')?.childElementCount).toBe(0);

    await interact(() =>
      document.querySelector<HTMLButtonElement>('button[aria-label="Retry loading the workflow and graph"]')?.click()
    );
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('"name":"clip"'));
    });
    expect(document.querySelector('[role="tab"][aria-selected="true"]')?.textContent).toBe('Workflow');
    expect(document.body.textContent).toContain('Could not load the metadata');
    expect(document.body.textContent).not.toContain('Could not load the workflow and graph');
  });

  it('keeps the video metadata tab when its workflow cannot be read, and retries only the workflow', async () => {
    galleryMocks.videoMetadata.mockResolvedValueOnce({ codec: 'h264' });
    galleryMocks.videoWorkflow
      .mockRejectedValueOnce(new Error('workflow unavailable'))
      .mockResolvedValueOnce({ graph: '{"nodes":[]}', workflow: '{"name":"clip"}' });

    await renderDetails({ isOpen: true, item: videoItem });
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('"codec": "h264"'));
    });

    expect(getButton('Workflow').getAttribute('aria-disabled')).toBe('true');
    expect(document.body.textContent).toContain('Could not load the workflow and graph');
    await interact(() =>
      document.querySelector<HTMLButtonElement>('button[aria-label="Retry loading the workflow and graph"]')?.click()
    );

    await act(async () => {
      await vi.waitFor(() => expect(getButton('Workflow').getAttribute('aria-disabled')).not.toBe('true'));
    });
    expect(galleryMocks.videoMetadata).toHaveBeenCalledOnce();
    expect(galleryMocks.videoWorkflow).toHaveBeenCalledTimes(2);
  });

  it('tells a failed image metadata read from one that recorded nothing', async () => {
    await renderDetails({ isOpen: true, item: imageItem });
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('queue-image'));
    });
    // Recorded nothing: an answer, not a failure.
    expect(document.querySelector('button[aria-label="Retry loading the metadata"]')).toBeNull();
    await interact(() => root?.unmount(), 0);
    queryClient.clear();
    root = createRoot(host!);

    galleryMocks.imageMetadata
      .mockRejectedValueOnce(new Error('metadata unavailable'))
      .mockResolvedValueOnce({ positive_prompt: 'read on retry' });
    await renderDetails({ isOpen: true, item: imageItem });
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('Could not load the metadata'));
    });

    await interact(() =>
      document.querySelector<HTMLButtonElement>('button[aria-label="Retry loading the metadata"]')?.click()
    );
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('read on retry'));
    });
    expect(document.body.textContent).not.toContain('Could not load the metadata');
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

    // Only the row buttons carry aria-labels; the verb-row buttons are named
    // by their visible text, so the attribute selector is unambiguous.
    const seedRecall = document.querySelector<HTMLButtonElement>('[aria-label="Use seed"]');
    expect(seedRecall).not.toBeNull();
    expect(document.querySelector('[aria-label="Use prompt"]')).not.toBeNull();
    // No size row was parsed, so no row carries the size verb even though the
    // capability itself is available.
    expect(document.querySelector('[aria-label="Use size"]')).toBeNull();

    // The model row has no single-field verb — copy stays its only action.
    const modelRow = [...(document.querySelectorAll('.chakra-data-list__item') ?? [])].find((row) =>
      row.textContent?.includes('Test Model')
    );
    expect(modelRow?.querySelectorAll('button')).toHaveLength(1);
    expect(modelRow?.querySelector('[aria-label="Copy"]')).not.toBeNull();

    await interact(() => seedRecall?.click());
    expect(actions.recallImageData).toHaveBeenCalledWith(actionImage, 'seed');
  });

  it('omits row recall buttons when the capability is unavailable', async () => {
    actions.deriveImageRecallCapabilities = vi.fn(() => NO_RECALL_CAPABILITIES);
    galleryMocks.imageMetadata.mockResolvedValueOnce({ seed: 42 });

    await renderDetails({ isOpen: true, item: imageItem });
    await act(async () => {
      await vi.waitFor(() => expect(document.body.textContent).toContain('42'));
    });

    expect(document.querySelector('[aria-label="Use seed"]')).toBeNull();
  });

  it('updates row recall verbs when capability inputs change without refetching cached metadata', async () => {
    galleryMocks.imageMetadata.mockResolvedValueOnce({ positive_prompt: 'cached prompt' });

    await renderDetails({ isOpen: true, item: imageItem });
    await act(async () => {
      await vi.waitFor(() => expect(document.querySelector('[aria-label="Use prompt"]')).not.toBeNull());
    });

    actions = {
      ...actions,
      deriveImageRecallCapabilities: vi.fn(() => NO_RECALL_CAPABILITIES),
    } as unknown as ImageActions;
    await renderDetails({ isOpen: true, item: imageItem });

    expect(document.querySelector('[aria-label="Use prompt"]')).toBeNull();
    expect(galleryMocks.imageMetadata).toHaveBeenCalledOnce();
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
