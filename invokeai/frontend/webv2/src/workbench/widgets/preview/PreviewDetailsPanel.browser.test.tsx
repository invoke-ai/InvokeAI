/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GalleryImage, GalleryImageItem, GalleryVideoItem } from '@features/gallery';
import type * as GalleryModule from '@features/gallery';
import type * as GalleryBackendModule from '@features/gallery/data/backend';
import type { ImageIndexAvailability } from '@features/gallery/data/backend';
import type * as GalleryQueriesModule from '@features/gallery/queries';
import type * as IdentityModule from '@features/identity';
import type { ImageActions } from '@workbench/image-actions';
import type { ImageMapImageLabels } from '@workbench/image-map/api';
import type { WidgetContributionSource } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createExtensionRegistry, type ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { FocusRegionProvider, useFocusRegionProps } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';
import { createInstance } from 'i18next';
import { act, createRef, useImperativeHandle, useState, type CSSProperties } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { PreviewDetailsPanel } from './PreviewDetailsPanel';

// Forty rows / a long payload: far taller than the stage below, so the cap is what keeps it inside.
const TALL_RECORD = Object.fromEntries(Array.from({ length: 40 }, (_, index) => [`field_${index}`, `value ${index}`]));

vi.mock('@features/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof GalleryModule>();

  return {
    ...actual,
    galleryImages: {
      ...actual.galleryImages,
      workflow: () => Promise.resolve({ graph: null, workflow: '{"name":"image workflow"}' }),
    },
    galleryVideos: {
      ...actual.galleryVideos,
      workflow: () => Promise.resolve({ graph: null, workflow: null }),
    },
  };
});
// The transport under the gallery's shared metadata queries.
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackendModule>()),
  getGalleryImageMetadata: () => Promise.resolve({ ...TALL_RECORD, positive_prompt: 'a tall prompt', seed: 7 }),
  getGalleryVideoMetadata: () => Promise.resolve(TALL_RECORD),
}));
const mocks = vi.hoisted(() => ({
  extensions: null as unknown as ExtensionRegistry,
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
// Preview is the center region's active widget; the workbench hotkey runtime routes keys by it.
const project = {
  id: 'project-1',
  widgetInstances: { 'preview-instance': { id: 'preview-instance', typeId: 'preview' } },
  widgetRegions: { center: { activeInstanceId: 'preview-instance', instanceIds: ['preview-instance'] } },
};
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectSelector: (selector: (value: typeof project) => unknown) => selector(project),
  useWidgetValuesSelector: (widgetId: string, selector: (values: Record<string, unknown>) => unknown) =>
    selector(
      (
        Object.values(project.widgetInstances).find((instance) => instance.typeId === widgetId) as
          | { state?: { values?: Record<string, unknown> } }
          | undefined
      )?.state?.values ?? {}
    ),
  useOptionalWorkbenchExtensions: () => mocks.extensions,
  useWorkbenchExtensions: () => mocks.extensions,
}));
// They need the whole application; the test registers Preview's step binding itself.
vi.mock('@workbench/hotkeys/firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));
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
const detailsToggle = createRef<() => void>();
const toggleDetails = () => detailsToggle.current?.();
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

/** CenterArea's chrome inset; its islands sit `top="2"`, `insetX="2"`, `h="8"` over the region. */
const CENTER_CHROME_INSET_STYLE = { '--wb-center-chrome-inset': '3rem' } as CSSProperties;

/**
 * A widget: the trigger in a side region's header row above a short stage, or in a centre chrome island floating over
 * it; a filmstrip row under the stage.
 */
const Widget = ({
  chrome,
  hasPanel,
  hasStage,
  item,
  stageKey,
}: {
  chrome: 'center' | 'side';
  hasPanel: boolean;
  hasStage: boolean;
  item: GalleryImageItem | GalleryVideoItem;
  stageKey: number;
}) => {
  const [stage, setStage] = useState<HTMLDivElement | null>(null);
  const [isOpen, setIsOpen] = useState(true);
  // What the Preview's I binding does through the widget value.
  useImperativeHandle(detailsToggle, () => () => setIsOpen((open) => !open), []);

  return (
    // The region container around the widget frame, as the shell nests them.
    <div {...useFocusRegionProps('center')}>
      <div
        data-hotkey-widget-instance-id="preview-instance"
        data-hotkey-widget-region="center"
        data-hotkey-widget-type-id="preview"
        style={{
          display: 'flex',
          flexDirection: 'column',
          height: 360,
          position: 'relative',
          width: 640,
          ...(chrome === 'center' ? CENTER_CHROME_INSET_STYLE : {}),
        }}
      >
        <Box
          data-testid="chrome"
          display="flex"
          h="8"
          justifyContent="flex-end"
          {...(chrome === 'center' ? { insetEnd: '2', position: 'absolute', top: '2', zIndex: '1' } : {})}
        >
          {hasPanel ? (
            <PreviewDetailsPanel
              actions={actions}
              image={item.kind === 'image' ? image : null}
              isOpen={isOpen}
              item={item}
              position={POSITION}
              stageElement={hasStage ? stage : null}
              onOpenChange={(open) => {
                onOpenChange(open);
                setIsOpen(open);
              }}
            />
          ) : null}
        </Box>
        {/* As Preview's stage is: positioned, so the panel pins to its corner. */}
        <div
          key={stageKey}
          ref={setStage}
          data-testid="stage"
          style={{ flex: 1, minHeight: 0, position: 'relative' }}
          tabIndex={-1}
        />
        <div data-testid="filmstrip" style={{ height: 60 }} tabIndex={-1} />
      </div>
    </div>
  );
};

const render = async (
  item: GalleryImageItem | GalleryVideoItem,
  {
    chrome = 'side',
    hasPanel = true,
    hasStage = true,
    stageKey = 0,
  }: { chrome?: 'center' | 'side'; hasPanel?: boolean; hasStage?: boolean; stageKey?: number } = {}
) => {
  await act(() =>
    root.render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <FocusRegionProvider controller={createTestFocusController()}>
              <WorkbenchHotkeyRuntime />
              {/* As a docked Preview sits in the workbench: the nearest region is the whole project. */}
              <div aria-label="Project" role="region">
                <Widget chrome={chrome} hasPanel={hasPanel} hasStage={hasStage} item={item} stageKey={stageKey} />
                <div
                  data-hotkey-widget-instance-id="gallery-instance"
                  data-hotkey-widget-region="left"
                  data-hotkey-widget-type-id="gallery"
                >
                  <button data-testid="gallery-tile" type="button">
                    Tile
                  </button>
                </div>
                <div
                  data-hotkey-widget-instance-id="canvas-instance"
                  data-hotkey-widget-region="right"
                  data-hotkey-widget-type-id="canvas"
                  data-testid="canvas"
                  style={{ height: 40 }}
                />
              </div>
              <button data-testid="outside" type="button">
                Outside
              </button>
            </FocusRegionProvider>
          </ChakraProvider>
        </I18nextProvider>
      </QueryClientProvider>
    )
  );
  await settle();
};

const panel = () => document.querySelector<HTMLElement>('[data-preview-details]');
const stage = () => host.querySelector<HTMLElement>('[data-testid="stage"]')!;
const filmstrip = () => host.querySelector<HTMLElement>('[data-testid="filmstrip"]')!;

/** The panel starts at the stage's top and ends on or above its bottom edge, never over the strip. */
const expectBoundedByStage = () => {
  const rect = panel()!.getBoundingClientRect();
  const stageRect = stage().getBoundingClientRect();

  expect(rect.top).toBeGreaterThanOrEqual(stageRect.top - 1);
  expect(rect.height).toBeGreaterThan(150);
  expect(rect.bottom).toBeLessThanOrEqual(stageRect.bottom + 0.5);
  expect(rect.bottom).toBeLessThanOrEqual(filmstrip().getBoundingClientRect().top + 0.5);
};

beforeEach(() => {
  mocks.extensions = createExtensionRegistry();
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

describe('PreviewDetailsPanel', () => {
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
    expect(tags!.getBoundingClientRect().right).toBeLessThanOrEqual(panel()!.getBoundingClientRect().right);
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

  it('moves between its tabs with the arrows without stepping the previewed item', async () => {
    // Preview's own step binding, as the widget registers it.
    const source: WidgetContributionSource = {
      instanceId: 'preview-instance',
      projectId: 'project-1',
      region: 'center',
      typeId: 'preview',
    };
    const step = vi.fn();
    mocks.extensions.commands.register({ handler: step, id: 'viewer.nextItem', source, title: 'Show next item' });
    mocks.extensions.hotkeys.register({
      commandId: 'viewer.nextItem',
      defaultKeys: ['arrowright'],
      id: 'viewer.nextItem',
      scope: 'widget',
      source,
      title: 'Show next item',
    });
    await render(imageItem);
    const press = async (target: HTMLElement) => {
      await act(async () => {
        target.focus();
        await userEvent.keyboard('{ArrowRight}');
      });
    };

    // On the stage the binding is live.
    await press(stage());
    expect(step).toHaveBeenCalledTimes(1);

    const detailsTab = document.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]')!;
    expect(detailsTab.textContent).toBe('Details');
    await press(detailsTab);
    // Focus moves to the next tab; selection waits for Enter, Space or a click.
    expect(document.activeElement).toBe(page.getByRole('tab', { name: 'Metadata' }).element());
    expect(step).toHaveBeenCalledTimes(1);
  });

  it('bounds video details by the stage too', async () => {
    await render(videoItem);

    await expect.element(page.getByRole('region', { name: 'Metadata JSON' })).toBeVisible();
    expectBoundedByStage();
    expect(document.body.textContent).toContain('1920 × 1080');
    await expect.element(page.getByRole('tab', { name: 'Workflow' })).toHaveAttribute('aria-disabled', 'true');
    await expect.element(page.getByRole('tab', { name: 'Graph' })).toHaveAttribute('aria-disabled', 'true');
  });

  it('pins a named region inside the stage, controlled by its trigger, without taking focus', async () => {
    const before = document.activeElement;
    await render(imageItem);

    await expect.element(page.getByText('a tall prompt')).toBeVisible();
    const region = page.getByRole('region', { name: 'Details' }).element();
    expect(region).toBe(panel());
    expect(region.parentElement).toBe(stage());
    expect(page.getByRole('button', { name: 'Details' }).element().getAttribute('aria-controls')).toBe(region.id);
    expect(document.activeElement).toBe(before);
    // Under a side region's header row: pinned to the stage's top-end corner.
    const rect = region.getBoundingClientRect();
    const stageRect = stage().getBoundingClientRect();
    expect(rect.top - stageRect.top).toBeCloseTo(8, 0);
    expect(stageRect.right - rect.right).toBeCloseTo(8, 0);
  });

  it('sits as far below the centre chrome island as the island sits below the stage top, sharing its end edge', async () => {
    await render(imageItem, { chrome: 'center' });
    await expect.element(page.getByText('a tall prompt')).toBeVisible();

    const rect = panel()!.getBoundingClientRect();
    const island = host.querySelector<HTMLElement>('[data-testid="chrome"]')!.getBoundingClientRect();
    const stageRect = stage().getBoundingClientRect();
    expect(island.top - stageRect.top).toBeGreaterThan(0);
    expect(Math.abs(rect.top - island.bottom - (island.top - stageRect.top))).toBeLessThanOrEqual(1);
    expect(Math.abs(rect.right - island.right)).toBeLessThanOrEqual(1);
    expect(stageRect.bottom - rect.bottom).toBeGreaterThanOrEqual(island.top - stageRect.top - 1);
  });

  it('stays open for interactions anywhere outside it', async () => {
    await render(imageItem);
    await expect.element(page.getByText('a tall prompt')).toBeVisible();

    // Real clicks: nothing outside is pointer-blocked, and focus is free to leave for each target.
    for (const testId of ['gallery-tile', 'canvas', 'outside', 'filmstrip']) {
      await page.getByTestId(testId).click();
    }
    expect(document.activeElement).toBe(filmstrip());
    await act(() => {
      document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'mouse' }));
      document.body.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerType: 'mouse' }));
    });

    expect(onOpenChange).not.toHaveBeenCalled();
    expect(panel()).not.toBeNull();
  });

  it('leaves Escape, unhandled, to whatever has focus outside it', async () => {
    await render(imageItem);
    await expect.element(page.getByText('a tall prompt')).toBeVisible();
    const tile = host.querySelector<HTMLElement>('[data-testid="gallery-tile"]')!;
    const prevented: boolean[] = [];
    tile.addEventListener('keydown', (event) => prevented.push(event.defaultPrevented));

    for (const target of [tile, host.querySelector<HTMLElement>('[data-testid="outside"]')!]) {
      await act(async () => {
        target.focus();
        await userEvent.keyboard('{Escape}');
      });
    }

    // The Canvas Escape ladder, among others, ignores a press already marked handled.
    expect(prevented).toEqual([false]);
    expect(onOpenChange).not.toHaveBeenCalled();
    expect(panel()).not.toBeNull();
  });

  it('closes on Escape from within, returning focus to its trigger', async () => {
    await render(imageItem);
    await expect.element(page.getByText('a tall prompt')).toBeVisible();

    await act(async () => {
      document.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]')!.focus();
      await userEvent.keyboard('{Escape}');
    });

    expect(onOpenChange).toHaveBeenCalledExactlyOnceWith(false);
    expect(panel()).toBeNull();
    expect(document.activeElement).toBe(page.getByRole('button', { name: 'Details' }).element());
  });

  it('toggles from its trigger', async () => {
    await render(imageItem);
    await expect.element(page.getByText('a tall prompt')).toBeVisible();
    const trigger = page.getByRole('button', { name: 'Details' });

    await trigger.click();
    expect(onOpenChange).toHaveBeenLastCalledWith(false);
    expect(panel()).toBeNull();
    await expect.element(trigger).toHaveAttribute('aria-pressed', 'false');
    expect(trigger.element().hasAttribute('aria-controls')).toBe(false);

    await trigger.click();
    await expect.element(page.getByText('a tall prompt')).toBeVisible();
    expect(onOpenChange).toHaveBeenLastCalledWith(true);
    await expect.element(trigger).toHaveAttribute('aria-pressed', 'true');
  });

  it('re-anchors in a remounted stage and renders nothing without one', async () => {
    await render(imageItem);
    await expect.element(page.getByText('a tall prompt')).toBeVisible();
    const firstStage = stage();

    await render(imageItem, { stageKey: 1 });
    expect(stage()).not.toBe(firstStage);
    expect(panel()?.parentElement).toBe(stage());

    await render(imageItem, { hasStage: false, stageKey: 1 });
    expect(panel()).toBeNull();
    await expect.element(page.getByRole('button', { name: 'Details' })).toHaveAttribute('aria-pressed', 'true');
    expect(onOpenChange).not.toHaveBeenCalled();
  });

  it('hands focus inside it to the trigger when the I binding closes it', async () => {
    const source: WidgetContributionSource = {
      instanceId: 'preview-instance',
      projectId: 'project-1',
      region: 'center',
      typeId: 'preview',
    };
    mocks.extensions.commands.register({
      handler: () => toggleDetails(),
      id: 'viewer.toggleMetadata',
      source,
      title: 'Toggle details',
    });
    mocks.extensions.hotkeys.register({
      commandId: 'viewer.toggleMetadata',
      defaultKeys: ['i'],
      id: 'viewer.toggleMetadata',
      scope: 'widget',
      source,
      title: 'Toggle details',
    });
    await render(imageItem);
    await expect.element(page.getByText('a tall prompt')).toBeVisible();

    await act(async () => {
      page.getByRole('tab', { name: 'Metadata' }).element().focus();
      await userEvent.keyboard('i');
    });

    await vi.waitFor(() => expect(panel()).toBeNull());
    expect(document.activeElement).toBe(page.getByRole('button', { name: 'Details' }).element());
  });

  it('hands focus inside it to the Preview region when its trigger leaves with it', async () => {
    await render(imageItem);
    await expect.element(page.getByText('a tall prompt')).toBeVisible();
    act(() => page.getByRole('tab', { name: 'Metadata' }).element().focus());

    await render(imageItem, { hasPanel: false });

    const region = host.querySelector<HTMLElement>('[data-focus-region="center"]')!;
    await vi.waitFor(() => expect(document.activeElement).toBe(region));
  });

  it('leaves focus that is already elsewhere when it closes', async () => {
    await render(imageItem);
    await expect.element(page.getByText('a tall prompt')).toBeVisible();
    const tile = host.querySelector<HTMLElement>('[data-testid="gallery-tile"]')!;
    act(() => tile.focus());

    await act(() => toggleDetails());
    await settle(50);

    expect(panel()).toBeNull();
    expect(document.activeElement).toBe(tile);
  });
});
