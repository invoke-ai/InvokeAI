/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { GalleryBoard } from '@features/gallery';

import { ChakraProvider, Menu, Portal } from '@chakra-ui/react';
import { getGallerySettings } from '@features/gallery/contracts';
import { galleryBoardsOptions, getGalleryListingBoardsQuery } from '@features/gallery/queries';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { SaveIcon } from 'lucide-react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { LayerContextAction } from './layerContextActions';

import { LayerSaveToBoardSubmenu } from './LayerSaveToBoardSubmenu';

vi.mock('@workbench/projects/library', () => ({
  useProjectLibrarySelector: (selector: (snapshot: { summaries: [] }) => unknown) => selector({ summaries: [] }),
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => 'project-1',
  useActiveProjectName: () => 'Project',
  // No stored gallery values: the panel's default listing.
  useActiveProjectSelector: () => ({}),
  useWidgetValuesSelector: () => ({}),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

const board = (id: string, actions: NonNullable<GalleryBoard['access']>['actions']): GalleryBoard => ({
  access: { actions, adminOverride: false, generalAccess: 'restricted', role: 'owner' },
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id,
  imageCount: 1,
  isInbox: false,
  kind: 'board',
  name: `Board ${id}`,
  ownerName: 'Me',
  projectId: null,
  videoCount: 0,
});

const saveAction = (isDisabled: boolean): LayerContextAction => ({
  defaultLabel: 'Save layer to board',
  handler: vi.fn(),
  icon: SaveIcon,
  id: 'save-to-board',
  isDisabled,
  labelKey: 'widgets.layers.actions.saveLayerToBoard',
  order: 10,
  picker: 'board',
  section: 'output',
});

let root: Root | null = null;
let host: HTMLDivElement | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

const interact = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

const settleUntil = async (isSettled: () => boolean, description: string, timeoutMs = 2000): Promise<void> => {
  const deadline = Date.now() + timeoutMs;
  while (!isSettled()) {
    if (Date.now() > deadline) {
      throw new Error(`Timed out after ${String(timeoutMs)}ms waiting for ${description}`);
    }
    await act(async () => {
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 10);
      });
    });
  }
};

/** zag treats a pointer move as a hover only when the position changes, so approach from a pixel away. */
const hover = async (target: HTMLElement): Promise<void> => {
  const rect = target.getBoundingClientRect();
  const clientY = rect.top + rect.height / 2;
  await interact(() => {
    for (const clientX of [rect.left + rect.width / 2 - 1, rect.left + rect.width / 2]) {
      target.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX, clientY, pointerType: 'mouse' }));
    }
  });
};

const render = async (action: LayerContextAction, onSelect = vi.fn()) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const listing = galleryBoardsOptions(getGalleryListingBoardsQuery(getGallerySettings({})));
  queryClient.setQueryData(listing.queryKey, [board('mine', ['read', 'add_media']), board('read-only', ['read'])]);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await interact(() => {
    root?.render(
      <QueryClientProvider client={queryClient}>
        <ChakraProvider value={system}>
          <Menu.Root open positioning={{ getAnchorRect: () => ({ height: 1, width: 1, x: 20, y: 20 }) }}>
            <Portal>
              <Menu.Positioner>
                <Menu.Content>
                  <Menu.Item value="before">Before</Menu.Item>
                  <LayerSaveToBoardSubmenu action={action} onSelect={onSelect} />
                  <Menu.Item value="after">After</Menu.Item>
                </Menu.Content>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>
        </ChakraProvider>
      </QueryClientProvider>
    );
  });

  const trigger = document.querySelector<HTMLButtonElement>('[aria-label="widgets.layers.actions.saveLayerToBoard"]');
  expect(trigger).not.toBeNull();
  return {
    listingQuery: () => queryClient.getQueryCache().find({ queryKey: listing.queryKey })!,
    onSelect,
    trigger: trigger!,
  };
};

/** Presses each key in the parent menu; returns the highlighted item's name (or value) after each press. */
const walkKeys = async (keys: readonly string[]): Promise<(string | undefined)[]> => {
  await interact(() => document.querySelector<HTMLElement>('[role="menu"]')!.focus());
  const highlighted: (string | undefined)[] = [];
  for (const key of keys) {
    await act(() => userEvent.keyboard(key));
    await interact(() => undefined);
    const item = document.querySelector<HTMLElement>('[data-highlighted]');
    highlighted.push(item?.getAttribute('aria-label') ?? item?.dataset.value);
  }
  return highlighted;
};

describe('LayerSaveToBoardSubmenu', () => {
  it('reads boards only once opened and saves to the board picked', async () => {
    const action = saveAction(false);
    const { listingQuery, onSelect, trigger } = await render(action);

    expect(listingQuery().getObserversCount()).toBe(0);

    await hover(trigger);
    await settleUntil(() => document.querySelector('[data-value="board-mine"]') !== null, 'the board list');

    expect(listingQuery().getObserversCount()).toBe(1);
    expect(document.querySelector('[data-value="board-none"]')?.textContent).toBe('widgets.gallery.uncategorized');
    // A board the caller can only read is no destination.
    expect(document.querySelector('[data-value="board-read-only"]')).toBeNull();

    await interact(() => document.querySelector<HTMLElement>('[data-value="board-mine"]')!.click());

    expect(onSelect).toHaveBeenCalledExactlyOnceWith(action, { id: 'mine', label: 'Board mine' });
  });

  it('offers no boards while the action is disabled', async () => {
    const { listingQuery, trigger } = await render(saveAction(true));

    expect(trigger.disabled).toBe(true);
    await hover(trigger);

    expect(document.querySelector('[data-value="board-none"]')).toBeNull();
    expect(listingQuery().getObserversCount()).toBe(0);
  });

  it('opens from the keyboard when enabled', async () => {
    await render(saveAction(false));

    expect(await walkKeys(['{ArrowDown}', '{ArrowDown}'])).toEqual([
      'before',
      'widgets.layers.actions.saveLayerToBoard',
    ]);
    await act(() => userEvent.keyboard('{ArrowRight}'));
    await settleUntil(() => document.querySelector('[data-value="board-none"]') !== null, 'the board list');
  });

  it('skips the disabled row from the keyboard, so ArrowRight cannot open it', async () => {
    const { listingQuery } = await render(saveAction(true));

    expect(await walkKeys(['{ArrowDown}', '{ArrowDown}'])).toEqual(['before', 'after']);
    await act(() => userEvent.keyboard('{ArrowRight}'));
    await interact(() => undefined);

    expect(document.querySelector('[data-value="board-none"]')).toBeNull();
    expect(listingQuery().getObserversCount()).toBe(0);
  });
});
