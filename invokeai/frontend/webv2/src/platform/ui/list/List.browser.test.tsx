import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ListContextMenuAnchor } from './ListItem';
import type { ListRow } from './listRows';

import { List, type ListRowProps } from './List';
import { ListItem } from './ListItem';
import { LIST_ROW_GAP_PX } from './listLayout';
import { listRowsFromSections } from './listRows';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) => (options?.name ? `${key}:${String(options.name)}` : key),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

interface Item {
  id: string;
  name: string;
}

const HEADER_PX = 32;
/** A comfortable row's slot: its 52px item plus the row gap. */
const ROW_PX = 52 + LIST_ROW_GAP_PX;

const section = (key: string, label: string, count: number): { items: Item[]; key: string; label: string } => ({
  items: Array.from({ length: count }, (_, index) => ({ id: `${key}-${index}`, name: `${label} ${index}` })),
  key,
  label,
});

const rowsOf = (...sections: { items: Item[]; key: string; label: string }[]): ListRow<Item>[] =>
  listRowsFromSections(sections, (item) => item.id);

const settleFrame = async () => {
  await act(async () => {
    await Promise.resolve();
  });
  await act(async () => {
    await new Promise((resolve) => {
      requestAnimationFrame(() => resolve(undefined));
    });
  });
};

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'display:flex;flex-direction:column;height:400px;width:360px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const Harness = ({
  activeKey = null,
  inset,
  rows,
  onCheckedChange,
  onContextMenu,
  onPress,
}: {
  activeKey?: string | null;
  inset?: 'row' | 'none';
  rows: ListRow<Item>[];
  onCheckedChange?: (item: Item, checked: boolean) => void;
  onContextMenu?: (item: Item, anchor: ListContextMenuAnchor) => void;
  onPress?: (item: Item) => void;
}) => (
  <ChakraProvider value={system}>
    <List
      activeKey={activeKey}
      density="comfortable"
      emptyState={<span>nothing</span>}
      inset={inset}
      label="subjects"
      renderItem={(item: Item, rowProps: ListRowProps) => (
        <ListItem
          {...rowProps}
          title={item.name}
          onCheckedChange={onCheckedChange ? (checked) => onCheckedChange(item, checked) : undefined}
          onContextMenu={onContextMenu ? (anchor) => onContextMenu(item, anchor) : undefined}
          onPress={onPress ? () => onPress(item) : undefined}
        />
      )}
      rows={rows}
    />
  </ChakraProvider>
);

const render = async (element: React.ReactElement) => {
  await act(() => root.render(element));
  await settleFrame();
};

const viewport = () => host.querySelector<HTMLElement>('[data-list-viewport]')!;
const pinnedHeader = () => host.querySelector<HTMLElement>('[data-list-pinned-header]');
const primaryButtons = () => [...host.querySelectorAll<HTMLButtonElement>('[data-list-primary]')];

const scrollTo = async (top: number) => {
  await act(async () => {
    viewport().scrollTop = top;
    viewport().dispatchEvent(new Event('scroll'));
    await new Promise((resolve) => {
      requestAnimationFrame(() => resolve(undefined));
    });
  });
};

describe('List virtualization and pinned headers', () => {
  it('mounts only the rows near the viewport and exposes the full set size', async () => {
    await render(<Harness rows={rowsOf(section('main', 'Main', 40), section('lora', 'LoRAs', 40))} />);

    const mounted = primaryButtons();

    expect(mounted.length).toBeGreaterThan(0);
    expect(mounted.length).toBeLessThan(40);
    expect(host.querySelector('[role="list"][aria-label="subjects"]')).not.toBeNull();
    expect(host.querySelector('[role="listitem"]')?.getAttribute('aria-setsize')).toBe('80');
    expect(host.querySelector('[role="listitem"]')?.getAttribute('aria-posinset')).toBe('1');
  });

  it('pins the header of the section at the top of the visible range and pushes it out with the next one', async () => {
    await render(<Harness rows={rowsOf(section('main', 'Main', 20), section('lora', 'LoRAs', 20))} />);

    expect(pinnedHeader()?.textContent).toContain('Main');
    expect(pinnedHeader()?.getBoundingClientRect().top).toBe(viewport().getBoundingClientRect().top);

    // The LoRAs header starts after the Main header and its 20 rows. One row past it, LoRAs is pinned.
    const loraHeaderTop = HEADER_PX + 20 * ROW_PX;

    await scrollTo(loraHeaderTop + ROW_PX);
    expect(pinnedHeader()?.textContent).toContain('LoRAs');
    expect(pinnedHeader()?.getBoundingClientRect().top).toBe(viewport().getBoundingClientRect().top);

    // Half a header before the LoRAs header arrives, the Main header is pushed half-way out.
    await scrollTo(loraHeaderTop - HEADER_PX / 2);
    expect(pinnedHeader()?.textContent).toContain('Main');
    expect(pinnedHeader()!.getBoundingClientRect().top).toBeCloseTo(
      viewport().getBoundingClientRect().top - HEADER_PX / 2,
      0
    );

    await scrollTo(0);
    expect(pinnedHeader()?.textContent).toContain('Main');
  });
});

describe('List active row', () => {
  it('opens scrolled to an active row far down the list so the selection is visible', async () => {
    await render(
      <ChakraProvider value={system}>
        <List
          activeKey="main-45"
          density="comfortable"
          label="subjects"
          renderItem={(item: Item, rowProps: ListRowProps) => <ListItem {...rowProps} title={item.name} />}
          revealActiveOnMount
          rows={rowsOf(section('main', 'Main', 60))}
        />
      </ChakraProvider>
    );

    const active = host.querySelector<HTMLElement>('[data-list-primary][aria-current="true"]')!;
    const viewportRect = viewport().getBoundingClientRect();
    const rowRect = active.getBoundingClientRect();

    expect(viewport().scrollTop).toBeGreaterThan(0);
    expect(rowRect.top).toBeGreaterThanOrEqual(viewportRect.top + HEADER_PX - 1);
    expect(rowRect.bottom).toBeLessThanOrEqual(viewportRect.bottom + 1);
  });
});

describe('List keyboard behavior', () => {
  it('rovers focus with the arrow keys, keeping one tab stop and following pointer focus', async () => {
    const onPress = vi.fn();

    await render(
      <Harness
        activeKey="main-2"
        rows={rowsOf(section('main', 'Main', 3), section('lora', 'LoRAs', 3))}
        onPress={onPress}
      />
    );

    const tabStops = primaryButtons().filter((button) => button.tabIndex === 0);

    expect(tabStops).toHaveLength(1);
    expect(tabStops[0]!.closest('[data-list-row]')?.getAttribute('data-list-row')).toBe('main-2');

    await act(() => tabStops[0]!.focus());
    await userEvent.keyboard('{ArrowDown}');
    await settleFrame();
    expect(document.activeElement?.closest('[data-list-row]')?.getAttribute('data-list-row')).toBe('lora-0');
    expect(primaryButtons().filter((button) => button.tabIndex === 0)).toHaveLength(1);

    await userEvent.keyboard('{End}');
    await settleFrame();
    expect(document.activeElement?.closest('[data-list-row]')?.getAttribute('data-list-row')).toBe('lora-2');

    await userEvent.keyboard('{Home}');
    await settleFrame();
    expect(document.activeElement?.closest('[data-list-row]')?.getAttribute('data-list-row')).toBe('main-0');

    await userEvent.keyboard('{Enter}');
    expect(onPress).toHaveBeenCalledWith({ id: 'main-0', name: 'Main 0' });

    const third = primaryButtons().find((button) => button.closest('[data-list-row="main-1"]'))!;

    await act(() => third.focus());
    expect(third.tabIndex).toBe(0);
    expect(primaryButtons().filter((button) => button.tabIndex === 0)).toHaveLength(1);
  });

  it('keeps the focused row mounted and scrolls it into view when arrowing far past the viewport', async () => {
    await render(<Harness rows={rowsOf(section('main', 'Main', 60))} />);

    await act(() => primaryButtons()[0]!.focus());
    await userEvent.keyboard('{End}');
    await settleFrame();
    await settleFrame();

    const focused = document.activeElement?.closest<HTMLElement>('[data-list-row]');

    expect(focused?.getAttribute('data-list-row')).toBe('main-59');
    expect(viewport().scrollTop).toBeGreaterThan(0);
    expect(focused!.getBoundingClientRect().bottom).toBeLessThanOrEqual(viewport().getBoundingClientRect().bottom + 1);
  });

  it('opens the context menu at the pointer, or under the row for keyboard requests, and can restore focus', async () => {
    const onContextMenu = vi.fn<(item: Item, anchor: ListContextMenuAnchor) => void>();

    await render(<Harness rows={rowsOf(section('main', 'Main', 3))} onContextMenu={onContextMenu} />);

    const row = host.querySelector<HTMLElement>('[data-list-row="main-1"]')!;

    await act(() => {
      row.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 90 }));
    });
    expect(onContextMenu).toHaveBeenLastCalledWith(
      { id: 'main-1', name: 'Main 1' },
      expect.objectContaining({ x: 120, y: 90 })
    );

    await act(() => {
      row.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 0, clientY: 0 }));
    });
    const rect = row.getBoundingClientRect();
    const anchor = onContextMenu.mock.lastCall![1];

    expect(anchor.x).toBeGreaterThanOrEqual(rect.left);
    expect(anchor.x).toBeLessThan(rect.right);
    expect(anchor.y).toBeCloseTo(rect.bottom, 0);

    (document.activeElement as HTMLElement | null)?.blur();
    await act(() => anchor.restoreFocus());
    expect(document.activeElement).toBe(row.querySelector('[data-list-primary]'));
  });
});

describe('List focus repair', () => {
  it('moves focus to the fallback row when the focused row disappears while the list owns focus', async () => {
    await render(<Harness activeKey="main-0" rows={rowsOf(section('main', 'Main', 4))} />);

    const second = primaryButtons().find((button) => button.closest('[data-list-row="main-1"]'))!;

    await act(() => second.focus());
    expect(document.activeElement).toBe(second);

    // A filter drops the focused row; focus must not be stranded on the body.
    await render(
      <Harness
        activeKey="main-0"
        rows={rowsOf({
          items: [
            { id: 'main-0', name: 'Main 0' },
            { id: 'main-3', name: 'Main 3' },
          ],
          key: 'main',
          label: 'Main',
        })}
      />
    );

    expect(document.activeElement?.closest('[data-list-row]')?.getAttribute('data-list-row')).toBe('main-0');
    expect(primaryButtons().filter((button) => button.tabIndex === 0)).toHaveLength(1);
  });
});

describe('List scroll offset persistence', () => {
  it('starts at the given offset and reports the last offset when it unmounts', async () => {
    const onScrollOffsetPersist = vi.fn();
    const rows = rowsOf(section('main', 'Main', 60));
    const element = (
      <ChakraProvider value={system}>
        <List
          density="comfortable"
          initialScrollOffset={500}
          label="subjects"
          renderItem={(item: Item, rowProps: ListRowProps) => <ListItem {...rowProps} title={item.name} />}
          rows={rows}
          onScrollOffsetPersist={onScrollOffsetPersist}
        />
      </ChakraProvider>
    );

    await render(element);
    expect(viewport().scrollTop).toBe(500);
    expect(host.querySelector('[data-index="0"]')).toBeNull();

    await scrollTo(1234);
    await act(() => root.unmount());
    root = createRoot(host);

    expect(onScrollOffsetPersist).toHaveBeenCalledTimes(1);
    expect(onScrollOffsetPersist).toHaveBeenCalledWith(1234);
  });
});

describe('a row editing its title inside a List', () => {
  const EditingHarness = ({ editingKey }: { editingKey: string | null }) => (
    <ChakraProvider value={system}>
      <button type="button">Before the list</button>
      <List
        density="comfortable"
        emptyState={<span>nothing</span>}
        label="subjects"
        renderItem={(item: Item, rowProps: ListRowProps) => (
          <ListItem
            {...rowProps}
            title={item.name}
            titleEditor={item.id === editingKey ? <input aria-label="Rename" defaultValue={item.name} /> : undefined}
            onPress={() => undefined}
          />
        )}
        rows={rowsOf(section('main', 'Main', 5))}
      />
    </ChakraProvider>
  );

  it('leaves the field its own Home, End and arrows, and gives the row its tab stop back after', async () => {
    await render(<EditingHarness editingKey={null} />);
    // The List's tab stop is on the second row, as after using it.
    await act(() => primaryButtons()[1]!.focus());

    await render(<EditingHarness editingKey="main-1" />);
    const field = host.querySelector<HTMLInputElement>('input[aria-label="Rename"]')!;
    await act(() => field.focus());
    field.setSelectionRange(2, 2);

    await act(() => userEvent.keyboard('{End}'));
    expect(field.selectionStart).toBe(field.value.length);
    await act(() => userEvent.keyboard('{Home}'));
    expect(field.selectionStart).toBe(0);
    await act(() => userEvent.keyboard('{ArrowDown}{ArrowUp}'));
    await settleFrame();
    expect(document.activeElement).toBe(field);

    // Editing done, the row's button is back and is the List's one tab stop.
    await render(<EditingHarness editingKey={null} />);
    const stops = primaryButtons().filter((button) => button.tabIndex === 0);
    expect(stops).toHaveLength(1);
    expect(stops[0]?.textContent).toContain('Main 1');

    await act(() => host.querySelector<HTMLButtonElement>('button:not([data-list-primary])')!.focus());
    await act(() => userEvent.tab());
    expect(document.activeElement).toBe(stops[0]);
  });
});
