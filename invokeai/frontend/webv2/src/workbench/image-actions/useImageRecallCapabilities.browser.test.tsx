import type { GalleryImage } from '@features/gallery';
import type * as GalleryBackendModule from '@features/gallery/data/backend';

import { fetchGalleryImageMetadata, galleryImageMetadataOptions } from '@features/gallery/queries';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ImageActions } from './useImageActions';

import { EMPTY_IMAGE_RECALL_CAPABILITIES, type ImageRecallCapabilities } from './imageRecall';
import { useImageRecallCapabilities } from './useImageRecallCapabilities';

const gallery = vi.hoisted(() => ({ metadata: vi.fn(), resolve: vi.fn() }));

// The transport, under both the gallery's shared metadata query and its public image port.
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackendModule>()),
  getGalleryImageByName: gallery.resolve,
  getGalleryImageMetadata: gallery.metadata,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Derive = ImageActions['deriveImageRecallCapabilities'];
type Result = ReturnType<typeof useImageRecallCapabilities>;

const IMAGE: GalleryImage = {
  boardId: 'none',
  height: 512,
  imageCategory: 'general',
  imageName: 'recall.png',
  imageUrl: '/full/recall.png',
  queuedAt: '2026-07-30T00:00:00.000Z',
  sourceQueueItemId: 'queue-recall',
  starred: false,
  thumbnailUrl: '/thumb/recall.png',
  width: 512,
};
const SEED_ONLY: ImageRecallCapabilities = { ...EMPTY_IMAGE_RECALL_CAPABILITIES, seed: true };

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient;
/** The hook's last answer, as the host printed it. */
const latest = (): Result | null => (host?.textContent ? (JSON.parse(host.textContent) as Result) : null);

const Host = ({ derive, image }: { derive: Derive; image: GalleryImage | null }) => {
  const result = useImageRecallCapabilities({ deriveImageRecallCapabilities: derive }, image);

  return <output>{JSON.stringify(result)}</output>;
};

const render = (derive: Derive, image: GalleryImage | null) =>
  act(() => {
    root?.render(
      <QueryClientProvider client={queryClient}>
        <Host derive={derive} image={image} />
      </QueryClientProvider>
    );
  });

/** Closes and reopens the menu: a new host over the same query cache. */
const remount = async (derive: Derive, image: GalleryImage | null) => {
  await act(() => root?.unmount());
  root = createRoot(host!);
  await render(derive, image);
};

/** Query results reach observers on a later task. */
const loaded = () => vi.waitFor(() => expect(latest()?.isLoading).toBe(false));

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });

  return { promise, resolve };
};

beforeEach(() => {
  accountLifecycle.activate('user-a');
  gallery.metadata.mockReset().mockResolvedValue({ seed: 7 });
  gallery.resolve.mockReset().mockResolvedValue({ ...IMAGE, hasWorkflow: true });
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  queryClient.clear();
  host?.remove();
  host = null;
  root = null;
});

describe('useImageRecallCapabilities', () => {
  it('derives from one read per image, and re-derives for a new Generate model without another', async () => {
    const derive = vi.fn<Derive>(() => SEED_ONLY);
    await render(derive, IMAGE);
    await loaded();

    expect(latest()).toEqual({ capabilities: SEED_ONLY, isLoading: false });
    // The listing did not say whether a workflow is embedded, so the record answered.
    expect(derive).toHaveBeenLastCalledWith({ ...IMAGE, hasWorkflow: true }, { seed: 7 });

    // A refreshed listing object for the same image reads nothing again.
    await render(derive, { ...IMAGE });

    const nextModel = vi.fn<Derive>(() => EMPTY_IMAGE_RECALL_CAPABILITIES);
    await render(nextModel, { ...IMAGE });

    expect(latest()).toEqual({ capabilities: EMPTY_IMAGE_RECALL_CAPABILITIES, isLoading: false });
    expect(nextModel).toHaveBeenLastCalledWith({ ...IMAGE, hasWorkflow: true }, { seed: 7 });
    // Neither fact changes for an image, so reopening the menu later reads nothing either.
    await remount(nextModel, IMAGE);
    expect(latest()).toEqual({ capabilities: EMPTY_IMAGE_RECALL_CAPABILITIES, isLoading: false });
    expect(gallery.metadata).toHaveBeenCalledOnce();
    expect(gallery.resolve).toHaveBeenCalledOnce();
  });

  it('reads metadata from the gallery entry other readers share', async () => {
    // Preview Details or a recall command read it first.
    await queryClient.fetchQuery(galleryImageMetadataOptions(captureAccountScope(), IMAGE.imageName));
    const derive = vi.fn<Derive>(() => SEED_ONLY);

    await render(derive, IMAGE);
    await loaded();

    expect(derive).toHaveBeenLastCalledWith({ ...IMAGE, hasWorkflow: true }, { seed: 7 });
    expect(gallery.metadata).toHaveBeenCalledOnce();
  });

  it('leaves a recall that joined its read to finish it when the menu closes first', async () => {
    gallery.metadata.mockImplementationOnce(
      (_name: string, signal: AbortSignal) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => reject(signal.reason));
        })
    );
    const derive = vi.fn<Derive>(() => SEED_ONLY);

    await render(derive, IMAGE);
    expect(gallery.metadata).toHaveBeenCalledOnce();
    const firstSignal = gallery.metadata.mock.calls[0]?.[1] as AbortSignal;
    // A recall picked from the open menu joins the menu's read.
    const recallRead = fetchGalleryImageMetadata(queryClient, captureAccountScope(), IMAGE.imageName);

    // The menu closes: its observer leaves and the shared read is cancelled.
    await act(() => root?.unmount());
    root = null;

    expect(firstSignal.aborted).toBe(true);
    await expect(recallRead).resolves.toEqual({ seed: 7 });
    expect(gallery.metadata).toHaveBeenCalledTimes(2);
  });

  it('reports loading until the read lands', async () => {
    const metadata = deferred<null>();
    gallery.metadata.mockReturnValueOnce(metadata.promise);
    const derive = vi.fn<Derive>(() => SEED_ONLY);

    await render(derive, IMAGE);

    expect(latest()).toEqual({ capabilities: EMPTY_IMAGE_RECALL_CAPABILITIES, isLoading: true });
    expect(derive).not.toHaveBeenCalled();

    metadata.resolve(null);
    await loaded();

    expect(latest()).toEqual({ capabilities: SEED_ONLY, isLoading: false });
  });

  it('derives from what the image carries when it no longer exists, and does not ask again', async () => {
    gallery.metadata.mockRejectedValueOnce(new ApiError('Not Found', 404));
    const derive = vi.fn<Derive>(() => SEED_ONLY);

    // The image already says it embeds a workflow, so its record is not read.
    await render(derive, { ...IMAGE, hasWorkflow: true });
    await loaded();

    expect(derive).toHaveBeenLastCalledWith({ ...IMAGE, hasWorkflow: true }, null);
    expect(gallery.resolve).not.toHaveBeenCalled();

    await remount(derive, { ...IMAGE, hasWorkflow: true });
    expect(gallery.metadata).toHaveBeenCalledOnce();
  });

  it('keeps the embedded workflow when metadata fails to load, and reads it again on the next open', async () => {
    gallery.metadata.mockRejectedValueOnce(new ApiError('Internal Server Error', 500));
    const derive = vi.fn<Derive>(() => SEED_ONLY);

    await render(derive, IMAGE);
    await loaded();

    // The record still answered, so the workflow (and what the image itself carries) stays on offer.
    expect(derive).toHaveBeenLastCalledWith({ ...IMAGE, hasWorkflow: true }, null);
    expect(latest()).toEqual({ capabilities: SEED_ONLY, isLoading: false });

    await remount(derive, IMAGE);
    await vi.waitFor(() => expect(gallery.metadata).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(derive).toHaveBeenLastCalledWith({ ...IMAGE, hasWorkflow: true }, { seed: 7 }));

    // Now complete, the answer is not read a third time.
    await remount(derive, IMAGE);
    expect(gallery.metadata).toHaveBeenCalledTimes(2);
  });

  it('treats a missing record as no embedded workflow', async () => {
    gallery.resolve.mockRejectedValueOnce(new ApiError('Not Found', 404));
    const derive = vi.fn<Derive>(() => SEED_ONLY);

    await render(derive, IMAGE);
    await loaded();

    expect(derive).toHaveBeenLastCalledWith({ ...IMAGE, hasWorkflow: false }, { seed: 7 });
  });

  it('offers no workflow while the record cannot be read, and reads it again on the next open', async () => {
    gallery.resolve.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    const derive = vi.fn<Derive>(() => SEED_ONLY);

    await render(derive, IMAGE);
    await loaded();

    expect(derive).toHaveBeenLastCalledWith({ ...IMAGE, hasWorkflow: false }, { seed: 7 });

    await remount(derive, IMAGE);
    await vi.waitFor(() => expect(derive).toHaveBeenLastCalledWith({ ...IMAGE, hasWorkflow: true }, { seed: 7 }));
    expect(gallery.resolve).toHaveBeenCalledTimes(2);
  });

  it('requests nothing for a null image', async () => {
    const derive = vi.fn<Derive>(() => SEED_ONLY);

    await render(derive, null);

    expect(latest()).toEqual({ capabilities: EMPTY_IMAGE_RECALL_CAPABILITIES, isLoading: false });
    expect(gallery.metadata).not.toHaveBeenCalled();
    expect(gallery.resolve).not.toHaveBeenCalled();
    expect(derive).not.toHaveBeenCalled();
  });

  it('cancels a read when the account changes; the remounted tree reads under the new account', async () => {
    const metadata = deferred<null>();
    gallery.metadata.mockReturnValueOnce(metadata.promise);
    const derive = vi.fn<Derive>(() => SEED_ONLY);

    await render(derive, IMAGE);
    const firstSignal = gallery.metadata.mock.calls[0]?.[1] as AbortSignal;

    await act(() => accountLifecycle.activate('user-b'));
    expect(firstSignal.aborted).toBe(true);

    // The authenticated tree is keyed by account epoch, so a new account mounts a new host.
    await remount(derive, IMAGE);
    await loaded();

    expect(gallery.metadata).toHaveBeenCalledTimes(2);
    expect(latest()).toEqual({ capabilities: SEED_ONLY, isLoading: false });
    metadata.resolve(null);
  });
});
