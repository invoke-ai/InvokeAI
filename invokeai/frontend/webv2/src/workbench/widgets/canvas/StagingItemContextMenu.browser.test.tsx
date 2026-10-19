import type { CanvasCandidateSlot } from '@workbench/canvasStagingView';
import type { ImageRecallCapabilities, ImageRecallKind } from '@workbench/image-actions';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { StagingItemContextMenu } from './StagingItemContextMenu';

const mocks = vi.hoisted(() => ({
  capabilities: {
    all: true,
    clipSkip: false,
    dimensions: true,
    prompts: true,
    remix: true,
    seed: false,
  } as ImageRecallCapabilities,
  meta: null as unknown,
  origin: null as string | null,
  recall: vi.fn<(kind: ImageRecallKind) => void>(),
}));

vi.mock('@features/queue/queries', () => ({
  getQueueReadModelOptions: (scope: { originPrefix: string }) => ({
    queryFn: () =>
      Promise.resolve({
        items: [
          {
            fieldValues: [
              { nodePath: 'positive_prompt', value: 'a staged cat' },
              { nodePath: 'seed', value: 4242 },
            ],
            id: 77,
          },
        ],
      }),
    queryKey: ['queue-read-model', scope.originPrefix],
  }),
}));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => 'project-1',
}));
vi.mock('@workbench/queue-integration/useQueueItemRecall', () => ({
  useQueueItemRecall: (origin: string, meta: unknown) => {
    mocks.origin = origin;
    mocks.meta = meta;
    return { capabilities: mocks.capabilities, isPending: false, recall: mocks.recall };
  },
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

const slot: CanvasCandidateSlot = {
  candidate: {
    height: 512,
    imageName: 'staged.png',
    imageUrl: '/staged.png',
    placement: { height: 512, opacity: 1, width: 512, x: 0, y: 0 },
    queuedAt: '2026-01-01T00:00:00.000Z',
    sourceBackendItemId: 77,
    sourceQueueItemId: 'local-queue-item',
    thumbnailUrl: '/staged-thumb.png',
    width: 512,
  },
  height: 512,
  id: 'slot-1',
  imageName: 'staged.png',
  kind: 'candidate',
  queueItemId: 'local-queue-item',
  width: 512,
};

const target = { slot, x: 40, y: 40 };

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

const menuItem = (label: string): HTMLElement => {
  const item = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (candidate) => candidate.textContent?.trim() === label
  );
  if (!item) {
    throw new Error(`Could not find menu item: ${label}`);
  }
  return item;
};

/** Real pointer events let the menu finish highlighting before selection; disabled rows remain inert under synthetic clicks too. */
const activate = async (label: string): Promise<void> => {
  if (menuItem(label).getAttribute('aria-disabled') !== 'true') {
    await act(() => page.getByRole('menuitem', { name: label, exact: true }).click());
    return;
  }
  await interact(() =>
    menuItem(label).dispatchEvent(new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse' }))
  );
  await interact(() => menuItem(label).click());
};

const render = async (handlers: Partial<Parameters<typeof StagingItemContextMenu>[0]> = {}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await interact(() =>
    root?.render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ChakraProvider value={system}>
          <StagingItemContextMenu
            canAccept
            target={target}
            onAccept={vi.fn()}
            onClose={vi.fn()}
            onDiscard={vi.fn()}
            onSaveToGallery={vi.fn()}
            {...handlers}
          />
        </ChakraProvider>
      </QueryClientProvider>
    )
  );
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  vi.clearAllMocks();
});

describe('StagingItemContextMenu', () => {
  it('recalls from the staged result’s own queue item and disables verbs its snapshot cannot back', async () => {
    await render();

    expect(mocks.origin).toBe('webv2:p:project-1:q:local-queue-item');
    // The executed prompt and seed come from the backend item that produced the result.
    await vi.waitFor(() => expect(mocks.meta).toEqual({ positivePrompt: 'a staged cat', seed: 4242 }));
    expect(menuItem('Use Seed').getAttribute('aria-disabled')).toBe('true');
    expect(menuItem('Use CLIP Skip').getAttribute('aria-disabled')).toBe('true');
    expect(menuItem('Recall All').getAttribute('aria-disabled')).toBeNull();

    await activate('Use Prompt');
    expect(mocks.recall).toHaveBeenCalledWith('prompts');

    // A greyed verb must stay inert: the row is only visually disabled.
    await activate('Use Seed');
    expect(mocks.recall).not.toHaveBeenCalledWith('seed');
  });

  it('keeps Accept inert while accepting is not allowed', async () => {
    const handlers = { canAccept: false, onAccept: vi.fn() };
    await render(handlers);

    expect(menuItem('widgets.canvas.acceptToLayer').getAttribute('aria-disabled')).toBe('true');
    await activate('widgets.canvas.acceptToLayer');
    expect(handlers.onAccept).not.toHaveBeenCalled();
  });

  it.each([
    ['widgets.canvas.staging.saveToGallery', 'onSaveToGallery'],
    ['widgets.canvas.acceptToLayer', 'onAccept'],
    ['common.discard', 'onDiscard'],
  ] as const)('%s runs the bar’s %s action for the candidate', async (label, handler) => {
    const handlers = { onAccept: vi.fn(), onDiscard: vi.fn(), onSaveToGallery: vi.fn() };
    await render(handlers);

    await activate(label);

    expect(handlers[handler]).toHaveBeenCalledOnce();
  });
});
