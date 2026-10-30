/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryItem } from '@features/gallery';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { FocusRegionProvider } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type * as PreviewHeaderStoreModule from './previewHeaderStore';

import { PreviewWidgetLabel } from './PreviewWidgetChrome';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          gallery: {
            commands: { clearSelection: 'Clear selection' },
            picker: { chooseMedia: 'Choose image or video' },
          },
          labels: { preview: 'Preview' },
          preview: { boardAndItem: '{{board}} / {{name}}', chooseAnotherItem: '{{label}}. Choose another item' },
        },
      },
    },
  },
});

const pickedItem: GalleryItem = {
  boardId: 'board-2',
  category: 'general',
  createdAt: '2026-09-01T00:00:00.000Z',
  durationSeconds: 4,
  fullUrl: '/full/clip.mp4',
  height: 64,
  isIntermediate: false,
  kind: 'video',
  name: 'clip.mp4',
  starred: false,
  thumbnailUrl: '/thumb/clip.mp4',
  width: 64,
};

/** The previewed item: an upload in a board the Gallery widget is not showing. */
const previewedItem: GalleryItem = {
  boardId: 'board-2',
  category: 'user',
  createdAt: '2026-09-02T00:00:00.000Z',
  fullUrl: '/full/image.png',
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name: 'image.png',
  starred: false,
  thumbnailUrl: '/thumb/image.png',
  width: 64,
};

const state = vi.hoisted(() => ({
  actionItem: null as GalleryItem | null,
  boardName: 'Fixture Board 02' as string | null,
  itemName: 'image.png' as string | null,
}));
const gallery = vi.hoisted(() => ({ clearSelection: vi.fn(), selectItemInItsBoard: vi.fn() }));
const pickerView = vi.hoisted(() => ({
  accept: null as null | readonly string[],
  openingScope: undefined as unknown,
}));

vi.mock('./previewHeaderStore', async (importOriginal) => ({
  ...(await importOriginal<typeof PreviewHeaderStoreModule>()),
  usePreviewHeaderContext: () => ({
    actionItem: state.actionItem,
    boardName: state.boardName,
    itemName: state.itemName,
  }),
}));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useWorkbenchCommands: () => ({ gallery }),
}));

// The picker's grid has its own tests; this stands in for it inside the real popover.
vi.mock('@features/gallery/ui/picker/GalleryPickerView', () => ({
  GalleryPickerView: ({
    accept,
    onClose,
    onPick,
    openingScope,
  }: {
    accept: readonly string[];
    onClose: () => void;
    onPick: (item: GalleryItem) => void;
    openingScope?: unknown;
  }) => {
    pickerView.accept = accept;
    pickerView.openingScope = openingScope;

    return (
      <button
        type="button"
        onClick={() => {
          onPick(pickedItem);
          onClose();
        }}
      >
        clip.mp4
      </button>
    );
  },
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const TRIGGER_NAME = 'Fixture Board 02 / image.png. Choose another item';

const render = async (region: 'center' | 'right' = 'center', width?: number) => {
  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={createTestFocusController()}>
            <div style={{ display: 'flex', width }}>
              <PreviewWidgetLabel region={region} />
            </div>
          </FocusRegionProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });
};

const getTrigger = (name = TRIGGER_NAME) => {
  const trigger = host?.querySelector<HTMLButtonElement>(`button[aria-label="${name}"]`);
  if (!trigger) {
    throw new Error('picker trigger did not render');
  }
  return trigger;
};

const getOpenDialog = () => document.querySelector<HTMLElement>('[role="dialog"][data-state="open"]');

beforeEach(() => {
  state.actionItem = previewedItem;
  state.boardName = 'Fixture Board 02';
  state.itemName = 'image.png';
  pickerView.accept = null;
  pickerView.openingScope = undefined;
  gallery.clearSelection.mockClear();
  gallery.selectItemInItsBoard.mockClear();
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

describe('PreviewWidgetLabel', () => {
  it('opens a mini gallery from the board and item name and selects the picked item', async () => {
    await render();
    const trigger = getTrigger();
    expect(trigger.textContent).toContain('Fixture Board 02');
    expect(trigger.textContent).toContain('image.png');

    await userEvent.click(trigger);
    await vi.waitFor(() => expect(getOpenDialog()?.getAttribute('aria-label')).toBe('Choose image or video'));
    const pick = await vi.waitFor(() => {
      const button = [...(getOpenDialog()?.querySelectorAll('button') ?? [])].find((b) => b.textContent === 'clip.mp4');
      expect(button).toBeDefined();
      return button!;
    });
    expect(pickerView.accept).toEqual(['image', 'video']);
    // At the previewed item's own board and, for an upload, the Assets view that lists it.
    expect(pickerView.openingScope).toEqual({ boardId: 'board-2', galleryView: 'assets' });

    await userEvent.click(pick);

    expect(gallery.selectItemInItsBoard).toHaveBeenCalledExactlyOnceWith(pickedItem);
    await vi.waitFor(() => expect(getOpenDialog()).toBeNull());
    await vi.waitFor(() => expect(document.activeElement).toBe(trigger));
  });

  it("opens a Canvas-owned item's picker at its board, leaving the view to the Gallery", async () => {
    state.actionItem = { ...previewedItem, category: 'other' };
    await render();

    await userEvent.click(getTrigger());
    await vi.waitFor(() => expect(pickerView.openingScope).toEqual({ boardId: 'board-2' }));
  });

  it('opens from the keyboard and returns focus to the trigger on Escape', async () => {
    await render('right');
    const trigger = getTrigger();
    trigger.focus();

    await userEvent.keyboard('{Enter}');
    await vi.waitFor(() => expect(getOpenDialog()).not.toBeNull());

    await userEvent.keyboard('{Escape}');
    await vi.waitFor(() => expect(getOpenDialog()).toBeNull());
    expect(document.activeElement).toBe(trigger);
    expect(gallery.selectItemInItsBoard).not.toHaveBeenCalled();
  });

  it('keeps the clear button beside the trigger', async () => {
    await render();
    const clear = host?.querySelector<HTMLButtonElement>('button[aria-label="Clear selection"]');

    await userEvent.click(clear!);

    expect(gallery.clearSelection).toHaveBeenCalledOnce();
    expect(getOpenDialog()).toBeNull();
  });

  it('gives way to the clear button at narrow widths, dropping the board before the item name truncates', async () => {
    const boardName = 'A Long Fixture Board Name';
    const itemName = 'a-very-long-generated-image-name-0123456789.png';
    state.boardName = boardName;
    state.itemName = itemName;
    const name = `${boardName} / ${itemName}. Choose another item`;
    const measure = async (width: number) => {
      await render('center', width);
      const trigger = getTrigger(name);
      const [item, board, slash] = [...trigger.querySelectorAll('p')];
      // The one-line row that clips whatever wraps below it.
      const row = item!.parentElement!.getBoundingClientRect();
      const visible = (element: Element | undefined) => {
        const rect = element?.getBoundingClientRect();
        return rect && rect.top < row.bottom && rect.bottom > row.top ? rect : null;
      };

      return {
        board: visible(board),
        clear: host!.querySelector('button[aria-label="Clear selection"]')!.getBoundingClientRect(),
        item: visible(item),
        slash: visible(slash),
        trigger: trigger.getBoundingClientRect(),
      };
    };

    const wide = await measure(640);
    const fullItemWidth = wide.item!.width;
    // Everything fits, in reading order: "Board / name".
    expect(wide.board!.right).toBeLessThanOrEqual(wide.slash!.left);
    expect(wide.slash!.right).toBeLessThanOrEqual(wide.item!.left);
    expect(wide.board!.left).toBeLessThan(wide.trigger.left + 16);

    // Tighter, the board name truncates while the item name keeps its full width.
    const boardTruncated = await measure(480);
    expect(boardTruncated.board!.width).toBeLessThan(wide.board!.width);
    expect(boardTruncated.item!.width).toBe(fullItemWidth);

    // With the board at its floor the name truncates too, but keeps a readable floor of its own.
    const bothAtFloor = await measure(300);
    expect(bothAtFloor.board!.width).toBeGreaterThan(0);
    expect(bothAtFloor.item!.width).toBeLessThan(fullItemWidth);
    expect(bothAtFloor.item!.width).toBeGreaterThan(bothAtFloor.board!.width * 2);

    for (const width of [200, 140]) {
      const narrow = await measure(width);
      // Side by side, never overlapping, with the clear button at its full size.
      expect(narrow.trigger.right).toBeLessThanOrEqual(narrow.clear.left);
      expect(narrow.clear.width).toBe(wide.clear.width);
      // "Board /" leaves together, never a bare "/"; the name stays, left-aligned in the trigger.
      expect(narrow.board).toBeNull();
      expect(narrow.slash).toBeNull();
      expect(narrow.item!.width).toBeGreaterThan(0);
      expect(narrow.item!.left).toBeLessThan(narrow.trigger.left + 16);
    }

    // Once the board drops, the name takes its room: wider at 200px than beside the board at 240px, still truncated.
    const alone = await measure(200);
    expect(alone.item!.width).toBeGreaterThan((await measure(240)).item!.width);
    expect(alone.item!.width).toBeLessThan(fullItemWidth);
  });

  it('shows the full name in a tooltip only while the label hides some of it', async () => {
    const getTooltip = () => document.querySelector('[role="tooltip"]');

    await render('center', 480);
    await userEvent.hover(getTrigger());
    // Give a wrongly gated tooltip time past its open delay.
    await new Promise((resolve) => {
      setTimeout(resolve, 800);
    });
    expect(getTooltip()).toBeNull();
    await userEvent.unhover(getTrigger());

    await render('center', 140);
    await userEvent.hover(getTrigger());
    await vi.waitFor(() => expect(getTooltip()?.textContent).toBe('Fixture Board 02 / image.png'));
  });

  it('renders no trigger without a selection', async () => {
    state.actionItem = null;
    state.boardName = null;
    state.itemName = null;
    await render('right');

    expect(host?.querySelector('button')).toBeNull();
    expect(host?.textContent).toBe('Preview');
  });
});
