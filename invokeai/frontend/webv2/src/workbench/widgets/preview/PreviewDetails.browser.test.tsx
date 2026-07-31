/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GalleryImage, GalleryImageItem, GalleryVideoItem } from '@features/gallery';
import type * as GalleryModule from '@features/gallery';
import type * as IdentityModule from '@features/identity';
import type { ImageActions, ImageRecallCapabilities } from '@workbench/image-actions';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PreviewFooter } from './PreviewFooter';

const galleryMocks = vi.hoisted(() => ({
  imageMetadata: vi.fn(),
  videoMetadata: vi.fn(),
  videoWorkflow: vi.fn(),
}));
const identityMocks = vi.hoisted(() => ({ accountEpoch: 7 }));

vi.mock('@features/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof GalleryModule>();

  return {
    ...actual,
    galleryImages: { ...actual.galleryImages, metadata: galleryMocks.imageMetadata },
    galleryVideos: {
      metadata: galleryMocks.videoMetadata,
      workflow: galleryMocks.videoWorkflow,
    },
  };
});
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
};
const NO_RECALL_CAPABILITIES: ImageRecallCapabilities = {
  all: false,
  clipSkip: false,
  dimensions: false,
  prompts: false,
  remix: false,
  seed: false,
};

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { countOfTotal: '{{count}} of {{total}}', copy: 'Copy' },
        widgets: {
          preview: {
            details: 'Details',
            framesPerSecond: '{{count}} fps',
            graph: 'Graph',
            graphJsonLabel: 'Video graph JSON',
            itemCount_one: '{{count}} item',
            itemCount_other: '{{count}} items',
            loadingMetadata: 'Loading metadata',
            metadata: 'Metadata',
            metadataJsonLabel: 'Video metadata JSON',
            nextItemInBoard: 'Next item in board',
            previousItemInBoard: 'Previous item in board',
            recallNotAvailable: 'Not recallable for this image',
            videoDuration: 'Duration {{duration}}',
            workflow: 'Workflow',
            workflowJsonLabel: 'Video workflow JSON',
          },
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
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

const renderFooter = async ({
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
            <PreviewFooter
              actionImage={item.kind === 'image' ? actionImage : null}
              actions={actions}
              boardItemCount={2}
              isLoadingBoard={false}
              isMetadataOpen={isOpen}
              item={item}
              selectedIndex={0}
              onNext={() => undefined}
              onPrevious={() => undefined}
              onToggleMetadata={() => undefined}
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
  galleryMocks.videoMetadata.mockReset().mockResolvedValue(null);
  galleryMocks.videoWorkflow.mockReset().mockResolvedValue({ graph: null, workflow: null });
  actions = {
    deriveImageRecallCapabilities: vi.fn(() => ALL_RECALL_CAPABILITIES),
    getImageRecallCapabilities: vi.fn(() => Promise.resolve(ALL_RECALL_CAPABILITIES)),
    recallImageData: vi.fn(() => Promise.resolve()),
  } as unknown as ImageActions;
  queryClient = new QueryClient({
    defaultOptions: { queries: { gcTime: Infinity, retry: false } },
  });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await interact(() => root?.unmount(), 0);
  queryClient.clear();
  host?.remove();
  host = null;
  root = null;
});

describe('Preview query-driven Details', () => {
  it('shows the video Details disclosure but starts no request while it is closed', async () => {
    await renderFooter({ isOpen: false, item: videoItem });

    expect(galleryMocks.videoMetadata).not.toHaveBeenCalled();
    expect(galleryMocks.videoWorkflow).not.toHaveBeenCalled();
    expect(getButton('Details')).not.toBeNull();
  });

  it('keeps the disabled image recall skeleton mounted while Details is pending', async () => {
    galleryMocks.imageMetadata.mockReturnValueOnce(
      new Promise(() => {
        // Intentionally pending.
      })
    );

    await renderFooter({ isOpen: true, item: imageItem });

    expect(host?.textContent).toContain('Loading metadata');
    expect(getButton('Recall All').disabled).toBe(true);
  });

  it('aborts the sole image metadata transport on close without starting duplicate capability work', async () => {
    const metadata = deferred<null>();
    galleryMocks.imageMetadata.mockReturnValueOnce(metadata.promise);

    await renderFooter({ isOpen: true, item: imageItem });

    const metadataSignal = galleryMocks.imageMetadata.mock.calls[0]?.[1] as AbortSignal;
    expect(actions.getImageRecallCapabilities).not.toHaveBeenCalled();

    await renderFooter({ isOpen: false, item: imageItem });

    expect(metadataSignal.aborted).toBe(true);
    await interact(() => {
      metadata.resolve(null);
    }, 0);
  });

  it('starts video metadata and workflow/graph concurrently under an epoch + GalleryItemKey query', async () => {
    const metadata = deferred<Record<string, unknown> | null>();
    const workflow = deferred<{ graph: string | null; workflow: string | null }>();
    galleryMocks.videoMetadata.mockReturnValueOnce(metadata.promise);
    galleryMocks.videoWorkflow.mockReturnValueOnce(workflow.promise);

    await renderFooter({ isOpen: true, item: videoItem });

    expect(galleryMocks.videoMetadata).toHaveBeenCalledOnce();
    expect(galleryMocks.videoWorkflow).toHaveBeenCalledOnce();
    expect(galleryMocks.videoMetadata.mock.calls[0]?.[0]).toBe('clip.mp4');
    expect(galleryMocks.videoWorkflow.mock.calls[0]?.[0]).toBe('clip.mp4');
    expect(galleryMocks.videoMetadata.mock.calls[0]?.[1]).toBeInstanceOf(AbortSignal);
    expect(galleryMocks.videoWorkflow.mock.calls[0]?.[1]).toBe(galleryMocks.videoMetadata.mock.calls[0]?.[1]);
    expect(queryClient.getQueryCache().find({ queryKey: ['preview', 'details', 7, 'video:clip.mp4'] })).toBeDefined();

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
    await renderFooter({ isOpen: true, item: videoItem });
    const metadataSignal = galleryMocks.videoMetadata.mock.calls[0]?.[1] as AbortSignal;
    const workflowSignal = galleryMocks.videoWorkflow.mock.calls[0]?.[1] as AbortSignal;

    await renderFooter({ isOpen: false, item: videoItem });

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
    await renderFooter({ isOpen: true, item: videoItem });
    const oldSignal = galleryMocks.videoMetadata.mock.calls[0]?.[1] as AbortSignal;
    const nextItem = {
      ...videoItem,
      fullUrl: '/videos/next.mp4',
      name: 'next.mp4',
      thumbnailUrl: '/thumbnails/next.webp',
    };

    await renderFooter({ isOpen: true, item: nextItem });

    expect(oldSignal.aborted).toBe(true);
    expect(galleryMocks.videoMetadata).toHaveBeenNthCalledWith(2, 'next.mp4', expect.any(AbortSignal));
    await act(async () => {
      await vi.waitFor(() => expect(host?.textContent).toContain('new-item'));
    });

    await interact(() => {
      oldMetadata.resolve({ marker: 'stale-item' });
      oldWorkflow.resolve({ graph: '{"marker":"stale-graph"}', workflow: '{"marker":"stale-workflow"}' });
    });
    expect(host?.textContent).not.toContain('stale-item');
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
    await renderFooter({ isOpen: true, item: videoItem });
    const oldSignal = galleryMocks.videoMetadata.mock.calls[0]?.[1] as AbortSignal;

    identityMocks.accountEpoch = 8;
    await renderFooter({ isOpen: true, item: videoItem });

    expect(oldSignal.aborted).toBe(true);
    expect(galleryMocks.videoMetadata).toHaveBeenCalledTimes(2);
    expect(queryClient.getQueryCache().find({ queryKey: ['preview', 'details', 8, 'video:clip.mp4'] })).toBeDefined();
    await act(async () => {
      await vi.waitFor(() => expect(host?.textContent).toContain('new-account'));
    });

    await interact(() => {
      oldMetadata.resolve({ marker: 'stale-account' });
      oldWorkflow.resolve({ graph: '{"epoch":7}', workflow: '{"epoch":7}' });
    });
    expect(host?.textContent).not.toContain('stale-account');
  });

  it('renders raw Metadata, Workflow, and Graph JSON tabs without image recall controls', async () => {
    galleryMocks.videoMetadata.mockResolvedValueOnce({ codec: 'h264', duration: 15 });
    galleryMocks.videoWorkflow.mockResolvedValueOnce({
      graph: '{"nodes":[{"id":"video"}]}',
      workflow: '{"name":"Video workflow"}',
    });

    await renderFooter({ isOpen: true, item: videoItem });

    await act(async () => {
      await vi.waitFor(() => expect(host?.textContent).toContain('"codec": "h264"'));
    });
    expect(getButton('Metadata')).not.toBeNull();
    expect(getButton('Workflow')).not.toBeNull();
    expect(getButton('Graph')).not.toBeNull();
    expect(host?.textContent).not.toContain('Recall All');

    await interact(() => getButton('Workflow').click());
    expect(host?.textContent).toContain('"name":"Video workflow"');
    await interact(() => getButton('Graph').click());
    expect(host?.textContent).toContain('"id":"video"');
    expect(host?.querySelector('[aria-label="Video graph JSON"]')).not.toBeNull();
  });

  it('keeps parsed image rows, source run, and recall controls while moving the fetch to Query', async () => {
    galleryMocks.imageMetadata.mockResolvedValueOnce({
      model: { name: 'Test Model' },
      positive_prompt: 'query-driven prompt',
      seed: 42,
    });

    await renderFooter({ isOpen: true, item: imageItem });

    expect(galleryMocks.imageMetadata).toHaveBeenCalledWith('still.png', expect.any(AbortSignal));
    expect(queryClient.getQueryCache().find({ queryKey: ['preview', 'details', 7, 'image:still.png'] })).toBeDefined();
    await act(async () => {
      await vi.waitFor(() => expect(host?.textContent).toContain('query-driven prompt'));
    });
    expect(host?.textContent).toContain('Test Model');
    expect(host?.textContent).toContain('queue-image');

    await interact(() => getButton('Recall All').click());
    expect(actions.recallImageData).toHaveBeenCalledWith(actionImage, 'all');
  });

  it('updates recall buttons when capability inputs change without refetching cached metadata', async () => {
    galleryMocks.imageMetadata.mockResolvedValueOnce({ positive_prompt: 'cached prompt' });

    await renderFooter({ isOpen: true, item: imageItem });
    await act(async () => {
      await vi.waitFor(() => expect(getButton('Recall All').disabled).toBe(false));
    });

    actions = {
      ...actions,
      deriveImageRecallCapabilities: vi.fn(() => NO_RECALL_CAPABILITIES),
    } as unknown as ImageActions;
    await renderFooter({ isOpen: true, item: imageItem });

    expect(getButton('Recall All').disabled).toBe(true);
    expect(galleryMocks.imageMetadata).toHaveBeenCalledOnce();
  });
});

const getButton = (text: string): HTMLButtonElement => {
  const button = Array.from((host ?? document).querySelectorAll<HTMLButtonElement>('button')).find(
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
