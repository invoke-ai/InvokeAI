/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { List, type ListRowProps } from './List';
import { ListItem } from './ListItem';
import { ListPager } from './ListPager';
import { listRowsFromItems } from './listRows';
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
});
