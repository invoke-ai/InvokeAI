/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { List, type ListRowProps } from './List';
import { ListItem } from './ListItem';
import { LIST_ROW_GAP_PX } from './listLayout';
import { ListPager } from './ListPager';
import { listRowsFromItems, listRowsFromSections } from './listRows';
import { ListSelectionBar } from './ListSelectionBar';
import { ListStack } from './ListStack';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) =>
      options && 'page' in options ? `${key}:${String(options.page)}` : key,
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'display:flex;flex-direction:column;height:300px;width:360px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const render = async (element: ReactElement) => {
  await act(() => root.render(<ChakraProvider value={system}>{element}</ChakraProvider>));
  await act(async () => {
    await Promise.resolve();
  });
};

const buttonWithText = (text: string): HTMLButtonElement =>
  [...host.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === text)!;

describe('ListPager', () => {
  it('keeps an unavailable control focusable and inert, and announces the page', async () => {
    const onNext = vi.fn();
    const onPrevious = vi.fn();

    await render(<ListPager hasNext hasPrevious={false} page={3} onNext={onNext} onPrevious={onPrevious} />);

    const previous = buttonWithText('common.previousPage');
    const next = buttonWithText('common.nextPage');

    expect(previous.getAttribute('aria-disabled')).toBe('true');
    expect(previous.disabled).toBe(false);
    expect(host.querySelector('[aria-live="polite"]')?.textContent).toBe('common.pageNumber:3');

    await act(() => previous.focus());
    await act(() => previous.click());
    expect(onPrevious).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(previous);

    await act(() => next.click());
    expect(onNext).toHaveBeenCalledTimes(1);

    await render(<ListPager isBusy hasNext hasPrevious page={3} onNext={onNext} onPrevious={onPrevious} />);
    await act(() => buttonWithText('common.nextPage').click());
    expect(onNext).toHaveBeenCalledTimes(1);
  });
});

describe('ListSelectionBar', () => {
  it('reflects a partial selection and shows actions only when given', async () => {
    const onCheckedChange = vi.fn();

    await render(
      <ListSelectionBar
        checked="indeterminate"
        label="Select all"
        summary="2 selected"
        onCheckedChange={onCheckedChange}
      >
        <button type="button">Delete</button>
      </ListSelectionBar>
    );

    const checkbox = host.querySelector<HTMLInputElement>('[aria-label="Select all"] input')!;

    expect(host.querySelector('[aria-label="Select all"]')?.getAttribute('data-state')).toBe('indeterminate');
    expect(host.textContent).toContain('2 selected');
    expect(buttonWithText('Delete')).toBeDefined();

    await act(() => checkbox.click());
    expect(onCheckedChange).toHaveBeenCalledTimes(1);

    await render(<ListSelectionBar checked={false} isDisabled label="Select all" onCheckedChange={onCheckedChange} />);
    expect(host.querySelector<HTMLInputElement>('[aria-label="Select all"] input')!.disabled).toBe(true);
    expect(host.querySelector('[role="separator"]')).toBeNull();
  });
});

describe('List busy state', () => {
  it('announces the list as busy and renders rows inert without dropping focus', async () => {
    const onPress = vi.fn();
    const onCheckedChange = vi.fn();
    const items = [{ id: 'a' }, { id: 'b' }];
    const rows = listRowsFromItems(items, (item) => item.id);
    const renderItem = (item: { id: string }, rowProps: ListRowProps) => (
      <ListItem {...rowProps} title={item.id} onCheckedChange={onCheckedChange} onPress={onPress} />
    );

    await render(<List label="subjects" renderItem={renderItem} rows={rows} />);
    const primary = host.querySelector<HTMLButtonElement>('[data-list-primary]')!;

    await act(() => primary.focus());
    await render(<List isBusy label="subjects" renderItem={renderItem} rows={rows} />);

    expect(host.querySelector('[role="list"]')?.getAttribute('aria-busy')).toBe('true');
    expect(document.activeElement).toBe(host.querySelector('[data-list-primary]'));
    expect(host.querySelector('[data-list-primary]')?.getAttribute('aria-disabled')).toBe('true');
    expect(host.querySelector('[data-list-row]')?.hasAttribute('data-busy')).toBe(true);
    expect(getComputedStyle(host.querySelector('[data-list-row]')!).opacity).toBe('1');
    expect(host.querySelector<HTMLInputElement>('[data-list-row] input')?.disabled).toBe(true);

    await act(() => host.querySelector<HTMLButtonElement>('[data-list-primary]')!.click());
    expect(onPress).not.toHaveBeenCalled();

    await render(<List label="subjects" renderItem={renderItem} rows={rows} />);
    expect(host.querySelector('[role="list"]')?.getAttribute('aria-busy')).toBeNull();
    await act(() => host.querySelector<HTMLButtonElement>('[data-list-primary]')!.click());
    expect(onPress).toHaveBeenCalledTimes(1);
  });
});

describe('list dividers', () => {
  const hairlines = () =>
    [...host.querySelectorAll<HTMLElement>('[aria-hidden="true"]')].filter(
      (element) => Math.round(element.getBoundingClientRect().height) === 1
    );

  it('draws an inset hairline centred in the gap between stacked rows, and none when off', async () => {
    await render(
      <ListStack dividers label="subjects">
        <ListItem title="one" />
        <ListItem title="two" />
        <ListItem title="three" />
      </ListStack>
    );

    const list = host.querySelector<HTMLElement>('[role="list"][aria-label="subjects"]')!;
    const rows = [...list.querySelectorAll<HTMLElement>(':scope > [role="listitem"]')];
    const lines = hairlines();

    expect(rows).toHaveLength(3);
    expect(lines).toHaveLength(2);

    const first = rows[0]!.getBoundingClientRect();
    const second = rows[1]!.getBoundingClientRect();
    const line = lines[0]!.getBoundingClientRect();

    // The row gap is unchanged by the divider, which sits in its middle.
    expect(second.top - first.bottom).toBeCloseTo(LIST_ROW_GAP_PX, 0);
    expect(line.top - first.bottom).toBeCloseTo((LIST_ROW_GAP_PX - 1) / 2, 0);
    // Inset from both edges so rounded row corners never touch it.
    expect(line.left - first.left).toBeCloseTo(8, 0);
    expect(first.right - line.right).toBeCloseTo(8, 0);

    await render(
      <ListStack label="subjects">
        <ListItem title="one" />
        <ListItem title="two" />
      </ListStack>
    );
    expect(hairlines()).toHaveLength(0);
  });

  it('draws dividers between virtualized rows of a section but not before a header or after the last row', async () => {
    const rows = listRowsFromSections(
      [
        { items: [{ id: 'a' }, { id: 'b' }], key: 'one', label: 'One' },
        { items: [{ id: 'c' }], key: 'two', label: 'Two' },
      ],
      (item) => item.id
    );

    await render(
      <List
        dividers
        label="subjects"
        renderItem={(item: { id: string }, rowProps: ListRowProps) => <ListItem {...rowProps} title={item.id} />}
        rows={rows}
      />
    );
    await act(async () => {
      await new Promise((resolve) => {
        requestAnimationFrame(() => resolve(undefined));
      });
    });

    // Only a → b shares a section with a following row.
    expect(hairlines()).toHaveLength(1);

    const rowA = host.querySelector<HTMLElement>('[data-list-row="a"]')!.getBoundingClientRect();
    const rowB = host.querySelector<HTMLElement>('[data-list-row="b"]')!.getBoundingClientRect();
    const line = hairlines()[0]!.getBoundingClientRect();

    // Same geometry as stacked rows: centred in the gap, inset from both row edges.
    expect(rowB.top - rowA.bottom).toBeCloseTo(LIST_ROW_GAP_PX, 0);
    expect(line.top - rowA.bottom).toBeCloseTo((LIST_ROW_GAP_PX - 1) / 2, 0);
    expect(line.left - rowA.left).toBeCloseTo(8, 0);
    expect(rowA.right - line.right).toBeCloseTo(8, 0);
  });

  it('keeps static rows and their neighbouring dividers unchanged on hover', async () => {
    await render(
      <ListStack dividers label="subjects">
        <ListItem title="one" />
        <div role="listitem">
          <ListItem role="presentation" title="two" />
        </div>
        <ListItem title="three" />
      </ListStack>
    );
    const row = host.querySelectorAll<HTMLElement>('[data-list-surface]')[1]!;
    const background = getComputedStyle(row).backgroundColor;

    await userEvent.hover(row);
    expect(row.querySelector('button')).toBeNull();
    expect(getComputedStyle(row).backgroundColor).toBe(background);
    expect(hairlines().map((line) => getComputedStyle(line).opacity)).toEqual(['1', '1']);
  });

  it('opens a menu-only row from keyboard, assistive-tech, and touch activation, but not a mouse click', async () => {
    const onContextMenu = vi.fn();
    await render(
      <ListStack label="subjects">
        <ListItem title="one" onContextMenu={onContextMenu} />
      </ListStack>
    );
    const primary = host.querySelector<HTMLButtonElement>('[data-list-primary]')!;
    const tap = (pointerType: string) => {
      primary.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType }));
      primary.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 }));
    };

    await act(() => tap('mouse'));
    expect(onContextMenu).not.toHaveBeenCalled();

    // NVDA/JAWS in Firefox click with detail 1 and no pointer press.
    await act(() => primary.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 })));
    await act(() => tap('touch'));
    await act(() => tap('pen'));
    expect(onContextMenu).toHaveBeenCalledTimes(3);
  });

  it('points a row whose only interaction is its actions', async () => {
    await render(
      <ListStack dividers label="subjects">
        <ListItem title="one" />
        <ListItem actions={<button type="button">Install</button>} title="two" />
        <ListItem title="three" />
      </ListStack>
    );
    const row = host.querySelectorAll<HTMLElement>('[data-list-surface]')[1]!;
    const background = getComputedStyle(row).backgroundColor;

    await userEvent.hover(row);
    // Still not a button itself: the install button is the row's only control.
    expect(row.querySelector('[data-list-primary]')?.tagName).toBe('DIV');
    await expect.poll(() => getComputedStyle(row).backgroundColor).not.toBe(background);
    await expect.poll(() => hairlines().map((line) => getComputedStyle(line).opacity)).toEqual(['0', '0']);
  });

  it('hides the hairlines either side of a pointed row, stacked or virtualized', async () => {
    const opacities = () =>
      [...host.querySelectorAll<HTMLElement>('[data-list-divider]')].map((line) => getComputedStyle(line).opacity);

    await render(
      <ListStack dividers label="subjects">
        <ListItem onPress={() => undefined} title="one" />
        <ListItem onPress={() => undefined} title="two" />
        <ListItem onPress={() => undefined} title="three" />
        <ListItem onPress={() => undefined} title="four" />
      </ListStack>
    );
    const stacked = host.querySelectorAll<HTMLElement>('[role="listitem"]');

    await userEvent.hover(stacked[1]!);
    await expect.poll(opacities).toEqual(['0', '0', '1']);
    await render(
      <ListStack dividers label="subjects">
        <ListItem onContextMenu={() => undefined} isMenuOpen title="one" />
        <ListItem onPress={() => undefined} title="two" />
        <ListItem onPress={() => undefined} title="three" />
        <ListItem onPress={() => undefined} title="four" />
      </ListStack>
    );
    await userEvent.unhover(host);
    await expect.poll(opacities).toEqual(['0', '1', '1']);

    // A caller-wrapped row (its own list item around the row) holding an open menu.
    await render(
      <ListStack dividers label="subjects">
        <ListItem onPress={() => undefined} title="one" />
        <div role="listitem">
          <ListItem onContextMenu={() => undefined} isMenuOpen role="presentation" title="two" />
        </div>
        <ListItem onPress={() => undefined} title="three" />
        <ListItem onPress={() => undefined} title="four" />
      </ListStack>
    );
    await expect.poll(opacities).toEqual(['0', '0', '1']);

    const rows = listRowsFromSections(
      [{ items: [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }], key: 'one', label: 'One' }],
      (item) => item.id
    );

    await render(
      <List
        dividers
        label="subjects"
        renderItem={(item: { id: string }, rowProps: ListRowProps) => <ListItem {...rowProps} title={item.id} />}
        rows={rows}
      />
    );
    await act(async () => {
      await new Promise((resolve) => {
        requestAnimationFrame(() => resolve(undefined));
      });
    });

    await userEvent.hover(host.querySelector<HTMLElement>('[data-list-row="c"]')!);
    await expect.poll(opacities).toEqual(['1', '0', '0']);
  });
});
