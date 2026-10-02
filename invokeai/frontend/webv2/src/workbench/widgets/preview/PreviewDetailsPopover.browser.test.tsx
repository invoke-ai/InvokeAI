/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GalleryImage, GalleryImageItem, GalleryVideoItem } from '@features/gallery';
import type * as GalleryModule from '@features/gallery';
import type { ImageIndexAvailability } from '@features/gallery/data/backend';
import type * as GalleryQueriesModule from '@features/gallery/queries';
import type * as IdentityModule from '@features/identity';
import type { ImageActions } from '@workbench/image-actions';
import type { ImageMapImageLabels } from '@workbench/image-map/api';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { PreviewDetailsPopover } from './PreviewDetailsPopover';

// Forty rows / a long payload: far taller than the stage below, so the cap is what keeps it inside.
const TALL_RECORD = Object.fromEntries(Array.from({ length: 40 }, (_, index) => [`field_${index}`, `value ${index}`]));

vi.mock('@features/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof GalleryModule>();

  return {
    ...actual,
    galleryImages: {
      ...actual.galleryImages,
      metadata: () => Promise.resolve({ ...TALL_RECORD, positive_prompt: 'a tall prompt', seed: 7 }),
      workflow: () => Promise.resolve({ graph: null, workflow: '{"name":"image workflow"}' }),
    },
    galleryVideos: {
      metadata: () => Promise.resolve(TALL_RECORD),
      workflow: () => Promise.resolve({ graph: null, workflow: null }),
    },
  };
});
const mocks = vi.hoisted(() => ({
  getImageLabels: vi.fn<() => Promise<ImageMapImageLabels | null>>(),
  indexAvailability: { modelName: null, state: 'disabled' } as ImageIndexAvailability,
}));

vi.mock('@features/gallery/queries', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryQueriesModule>()),
  imageIndexAvailabilityOptions: () => ({
    queryFn: () => mocks.indexAvailability,
    queryKey: ['test-image-index-availability'],
  }),
}));
vi.mock('@workbench/image-map/imageLabelCache', () => ({ getImageLabels: mocks.getImageLabels }));
vi.mock('@features/identity', async (importOriginal) => {
  const actual = await importOriginal<typeof IdentityModule>();

  return { ...actual, useAuthSession: () => ({ accountEpoch: 1 }) };
});

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
            graph: 'Graph',
            graphJsonLabel: 'Graph JSON',
            imageTags: 'Image tags:',
            loadingMetadata: 'Loading metadata',
            metadata: 'Metadata',
            metadataJsonLabel: 'Metadata JSON',
            videoDuration: 'Duration {{duration}}',
            workflow: 'Workflow',
            workflowJsonLabel: 'Workflow JSON',
          },
        },
      },
    },
  },
});

const imageItem: GalleryImageItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-09-21T00:00:00Z',
  fullUrl: '/images/still.png',
  height: 768,
  isIntermediate: false,
  kind: 'image',
  name: 'still.png',
  starred: false,
  thumbnailUrl: '/thumbnails/still.webp',
  width: 512,
};
const image: GalleryImage = {
  boardId: 'none',
  height: 768,
  imageCategory: 'general',
  imageName: 'still.png',
  imageUrl: '/images/still.png',
  queuedAt: '2026-09-21T00:00:00Z',
  sourceQueueItemId: 'run-1',
  starred: false,
  thumbnailUrl: '/thumbnails/still.webp',
  width: 512,
};
const videoItem: GalleryVideoItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-09-21T00:00:00Z',
  durationSeconds: 5,
  fullUrl: '/videos/clip.mp4',
  height: 1080,
  isIntermediate: false,
  kind: 'video',
  name: 'clip.mp4',
  starred: false,
  thumbnailUrl: '/thumbnails/clip.webp',
  width: 1920,
};
const actions = {
  deriveImageRecallCapabilities: () => ({
    all: true,
    clipSkip: false,
    dimensions: false,
    prompts: true,
    remix: true,
    seed: true,
    workflow: false,
  }),
  recallImageData: vi.fn(),
} as unknown as ImageActions;
const POSITION = { boardItemCount: 3, isLoadingBoard: false, selectedIndex: 0 };

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;
let queryClient: QueryClient;
const onOpenChange = vi.fn();

const settle = (ms = 300) =>
  act(
    () =>
      new Promise<void>((resolve) => {
        setTimeout(resolve, ms);
      })
  );

/** A widget: a header row holding the trigger, a short stage, and a filmstrip row under it. */
const Widget = ({ item }: { item: GalleryImageItem | GalleryVideoItem }) => {
  const [stage, setStage] = useState<HTMLDivElement | null>(null);

  return (
    <div role="region" style={{ display: 'flex', flexDirection: 'column', height: 360, width: 640 }}>
      <div style={{ display: 'flex', height: 32, justifyContent: 'flex-end' }}>
        <PreviewDetailsPopover
          actions={actions}
          image={item.kind === 'image' ? image : null}
          isOpen
          item={item}
          position={POSITION}
          stageElement={stage}
          onOpenChange={onOpenChange}
        />
      </div>
      <div ref={setStage} data-testid="stage" style={{ flex: 1, minHeight: 0 }} />
      <div data-testid="filmstrip" style={{ height: 60 }} />
    </div>
  );
};

const render = async (item: GalleryImageItem | GalleryVideoItem) => {
  await act(() =>
    root.render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <Widget item={item} />
          </ChakraProvider>
        </I18nextProvider>
      </QueryClientProvider>
    )
  );
  await settle();
};

const popover = () => document.querySelector<HTMLElement>('[data-preview-details]')!;
const stage = () => host.querySelector<HTMLElement>('[data-testid="stage"]')!;
const filmstrip = () => host.querySelector<HTMLElement>('[data-testid="filmstrip"]')!;

/** The popover sits under its trigger and ends on or above the stage's bottom edge, never over the strip. */
const expectBoundedByStage = () => {
  const rect = popover().getBoundingClientRect();
  const stageRect = stage().getBoundingClientRect();

  expect(rect.top).toBeGreaterThanOrEqual(stageRect.top - 1);
  expect(rect.height).toBeGreaterThan(150);
  expect(rect.bottom).toBeLessThanOrEqual(stageRect.bottom + 0.5);
  expect(rect.bottom).toBeLessThanOrEqual(filmstrip().getBoundingClientRect().top + 0.5);
};

beforeEach(() => {
  onOpenChange.mockReset();
  mocks.getImageLabels.mockReset();
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  mocks.indexAvailability = { modelName: null, state: 'disabled' };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('PreviewDetailsPopover', () => {
  it('keeps long image details inside the stage, scrolling them above the filmstrip', async () => {
    await render(imageItem);

    await expect.element(page.getByText('a tall prompt')).toBeVisible();
    expectBoundedByStage();
    expect(document.body.textContent).toContain('1 of 3');
    expect(document.body.textContent).toContain('512 × 768');

    // Workflow-added keys are not parsed rows; they live in the raw record, whose
    // forty lines overflow the cap and scroll inside it rather than past it.
    await page.getByRole('tab', { name: 'Metadata' }).click();
    const viewport = page.getByRole('region', { name: 'Metadata JSON' }).element();
    await expect.element(page.getByRole('region', { name: 'Metadata JSON' })).toBeVisible();
    expectBoundedByStage();
    expect(viewport.scrollHeight).toBeGreaterThan(viewport.clientHeight + 40);
  });

  it('puts the image tags after the dimensions, truncating them rather than the dimensions', async () => {
    mocks.indexAvailability = { modelName: null, state: 'ready' };
    mocks.getImageLabels.mockResolvedValue({
      alternates: ['a rocky shoreline beneath dramatic evening clouds', 'waves breaking over weathered stones'],
      label: 'sunset over a coastal lighthouse',
    });
    await render(imageItem);

    // Availability, the lazily loaded cache, and the label request resolve in turn; wait rather than time them.
    const tags = await vi.waitFor(() => {
      const element = document.querySelector<HTMLElement>('[data-preview-image-tags]');
      expect(element).not.toBeNull();
      return element;
    });
    const dimensions = page.getByText('512 × 768').element();
    expect(mocks.getImageLabels).toHaveBeenCalledWith({ kind: 'image', name: 'still.png' });
    expect(tags?.textContent).toBe(
      '·Image tags: sunset over a coastal lighthouse, a rocky shoreline beneath dramatic evening clouds, waves breaking over weathered stones'
    );
    // Same header row, to the right of the dimensions, which stay whole while the tags end in an ellipsis.
    expect(tags!.getBoundingClientRect().left).toBeGreaterThan(dimensions.getBoundingClientRect().right);
    expect(Math.abs(tags!.getBoundingClientRect().top - dimensions.getBoundingClientRect().top)).toBeLessThan(2);
    expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
    expect(tags!.scrollWidth).toBeGreaterThan(tags!.clientWidth);
    expect(getComputedStyle(tags!).textOverflow).toBe('ellipsis');
    expect(tags!.getBoundingClientRect().right).toBeLessThanOrEqual(popover().getBoundingClientRect().right);
  });

  it('withdraws the tags when the image index stops being ready', async () => {
    mocks.indexAvailability = { modelName: null, state: 'ready' };
    mocks.getImageLabels.mockResolvedValue({ alternates: [], label: 'sunset' });
    await render(imageItem);
    await vi.waitFor(() =>
      expect(document.querySelector('[data-preview-image-tags]')?.textContent).toBe('·Image tags: sunset')
    );

    await act(() => {
      queryClient.setQueryData(['test-image-index-availability'], { modelName: 'clip', state: 'model_missing' });
    });

    expect(document.querySelector('[data-preview-image-tags]')).toBeNull();
    expect(document.body.textContent).toContain('512 × 768');
  });

  it('leaves no separator behind for an item without tags', async () => {
    mocks.indexAvailability = { modelName: null, state: 'ready' };
    mocks.getImageLabels.mockResolvedValue(null);
    await render(imageItem);

    await vi.waitFor(() => expect(mocks.getImageLabels).toHaveBeenCalled());
    await settle(50);
    expect(document.querySelector('[data-preview-image-tags]')).toBeNull();
    // The header row ends at the dimensions.
    expect(page.getByText('512 × 768').element().parentElement?.textContent).toMatch(/512 × 768$/);
  });

  it('shows no tags, and asks for none, while the image index is not ready', async () => {
    mocks.getImageLabels.mockResolvedValue({ alternates: [], label: 'sunset' });
    await render(imageItem);

    expect(document.body.textContent).toContain('512 × 768');
    expect(document.querySelector('[data-preview-image-tags]')).toBeNull();
    expect(mocks.getImageLabels).not.toHaveBeenCalled();
  });

  it('shows the raw metadata record and the workflow for an image, like the legacy viewer', async () => {
    await render(imageItem);

    await page.getByRole('tab', { name: 'Metadata' }).click();
    await expect.element(page.getByRole('region', { name: 'Metadata JSON' })).toBeVisible();
    expect(document.body.textContent).toContain('"field_39"');
    expectBoundedByStage();

    await page.getByRole('tab', { name: 'Workflow' }).click();
    await expect.element(page.getByRole('region', { name: 'Workflow JSON' })).toBeVisible();
    expect(document.body.textContent).toContain('image workflow');
    // Nothing behind Graph, so it is not offered.
    await expect.element(page.getByRole('tab', { name: 'Graph' })).toHaveAttribute('aria-disabled', 'true');
  });

  it('bounds video details by the stage too', async () => {
    await render(videoItem);

    await expect.element(page.getByRole('region', { name: 'Metadata JSON' })).toBeVisible();
    expectBoundedByStage();
    expect(document.body.textContent).toContain('1920 × 1080');
    await expect.element(page.getByRole('tab', { name: 'Workflow' })).toHaveAttribute('aria-disabled', 'true');
    await expect.element(page.getByRole('tab', { name: 'Graph' })).toHaveAttribute('aria-disabled', 'true');
  });

  it('stays open for interactions inside the preview widget and does not steal focus', async () => {
    const before = document.activeElement;
    await render(imageItem);

    await expect.element(page.getByText('a tall prompt')).toBeVisible();
    expect(popover().contains(document.activeElement)).toBe(false);
    expect(document.activeElement).toBe(before);

    await act(() => {
      filmstrip().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'mouse' }));
      filmstrip().dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerType: 'mouse' }));
      filmstrip().dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    await settle(100);
    expect(onOpenChange).not.toHaveBeenCalled();

    await act(() => {
      document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'mouse' }));
      document.body.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerType: 'mouse' }));
    });
    await settle(100);
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});
