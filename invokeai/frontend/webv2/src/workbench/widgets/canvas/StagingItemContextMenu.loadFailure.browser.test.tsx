/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as MenuModule from '@platform/ui/Menu';
import type { CanvasCandidateSlot } from '@workbench/canvasStagingView';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

// The verbs' import fails as a dropped connection would until `fail` clears. Its own file: a page keeps loaded verbs.
const load = vi.hoisted(() => ({ fail: true }));
vi.mock('@platform/ui/Menu', async (importOriginal) => {
  const original = await importOriginal<typeof MenuModule>();

  return {
    ...original,
    createMenuPartLoader: <Part,>(importPart: () => Promise<Part>) =>
      original.createMenuPartLoader(() =>
        load.fail ? Promise.reject(new TypeError('Failed to fetch dynamically imported module')) : importPart()
      ),
  };
});
vi.mock('@features/queue/queries', () => ({
  getQueueReadModelOptions: () => ({ queryFn: () => Promise.resolve({ items: [] }), queryKey: ['queue-read-model'] }),
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => 'project-1',
}));
vi.mock('@workbench/queue-integration/useQueueItemRecall', () => ({
  useQueueItemRecall: () => ({
    capabilities: { all: true, clipSkip: true, dimensions: true, prompts: true, remix: true, seed: true },
    isPending: false,
    reasons: {},
    recall: vi.fn(),
  }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { StagingItemContextMenu } from './StagingItemContextMenu';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const slot = {
  candidate: { imageName: 'staged.png', sourceBackendItemId: 77, sourceQueueItemId: 'local-queue-item' },
  id: 'slot-1',
  kind: 'candidate',
  queueItemId: 'local-queue-item',
} as CanvasCandidateSlot;

const host = document.createElement('div');
const root = createRoot(host);
const unhandled = vi.fn();

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  window.removeEventListener('unhandledrejection', unhandled);
});

/** Opens the menu, which loads its verbs as it mounts. */
const open = async (onSaveToGallery = vi.fn()) => {
  await act(() =>
    root.render(
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <StagingItemContextMenu
            canAccept
            target={{ slot, x: 40, y: 40 }}
            onAccept={vi.fn()}
            onClose={vi.fn()}
            onDiscard={vi.fn()}
            onSaveToGallery={onSaveToGallery}
          />
        </ChakraProvider>
      </QueryClientProvider>
    )
  );
};

it('says the verbs could not load, keeps the rest of the menu working, and loads them on a later open', async () => {
  window.addEventListener('unhandledrejection', unhandled);
  document.body.append(host);
  const onSaveToGallery = vi.fn();
  await open(onSaveToGallery);

  const failed = page.getByRole('menuitem', { name: 'common.menuPartLoadFailed' });
  await expect.element(failed).toHaveAttribute('aria-disabled', 'true');
  expect(page.getByRole('menu').element().getAttribute('aria-busy')).toBe('false');
  await page.getByRole('menuitem', { name: 'widgets.canvas.staging.saveToGallery' }).click();
  expect(onSaveToGallery).toHaveBeenCalledOnce();

  await act(() => root.render(null));
  load.fail = false;
  await open();

  await expect
    .element(page.getByRole('menuitem', { name: 'widgets.gallery.itemActions.recall.verbs.all' }))
    .toBeVisible();
  expect(failed.query()).toBeNull();
  expect(unhandled).not.toHaveBeenCalled();
});
