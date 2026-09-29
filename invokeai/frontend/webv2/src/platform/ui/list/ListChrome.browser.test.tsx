/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { List, type ListRowProps } from './List';
import { ListItem } from './ListItem';
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
    expect(second.top - first.bottom).toBeCloseTo(4, 0);
    expect(line.top - first.bottom).toBeCloseTo(1.5, 0);
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
    expect(rowB.top - rowA.bottom).toBeCloseTo(4, 0);
    expect(line.top - rowA.bottom).toBeCloseTo(1.5, 0);
    expect(line.left - rowA.left).toBeCloseTo(8, 0);
    expect(rowA.right - line.right).toBeCloseTo(8, 0);
  });
});
