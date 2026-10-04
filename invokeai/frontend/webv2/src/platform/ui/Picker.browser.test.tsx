import { ChakraProvider } from '@chakra-ui/react';
import { auditAccessibility, contrastOffenderTexts } from '@platform/browser/auditAccessibility.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { PickerGroup, PickerStatus } from './Picker';

import { Picker } from './Picker';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

interface Fruit {
  detail?: string;
  disabled?: boolean;
  id: string;
  name: string;
}

const FRUIT_GROUPS: PickerGroup<Fruit>[] = [
  {
    colorPalette: 'orange',
    id: 'citrus',
    name: 'Citrus',
    options: [
      { id: 'lemon', name: 'Lemon' },
      { disabled: true, id: 'lime', name: 'Lime' },
      { id: 'orange', name: 'Orange' },
    ],
  },
  { colorPalette: 'red', id: 'berries', name: 'Berries', options: [{ id: 'strawberry', name: 'Strawberry' }] },
];
const MANY_GROUPS: PickerGroup<Fruit>[] = [
  {
    id: 'all',
    name: 'All',
    options: Array.from({ length: 500 }, (_, index) => ({ id: `fruit ${index}`, name: `Fruit ${index}` })),
  },
];

// Four colored groups of 150, every third option two lines tall, so estimates and measured heights disagree.
const DEEP_GROUPS: PickerGroup<Fruit>[] = ['Apples', 'Pears', 'Plums', 'Grapes'].map((name, groupIndex) => ({
  colorPalette: ['green', 'orange', 'purple', 'red'][groupIndex],
  id: name.toLowerCase(),
  name,
  options: Array.from({ length: 150 }, (_, index) => ({
    detail: index % 3 === 0 ? `${name} detail ${index}` : undefined,
    id: `${name.toLowerCase()}-${index}`,
    name: `${name.slice(0, -1)} ${index}`,
  })),
}));

const getOptionId = (fruit: Fruit) => fruit.id;
const getIsOptionDisabled = (fruit: Fruit) => fruit.disabled === true;
const isMatch = (fruit: Fruit, term: string) => fruit.name.toLowerCase().includes(term.toLowerCase());
const renderOption = (fruit: Fruit) =>
  fruit.detail ? (
    <>
      {fruit.name}
      <div style={{ height: 30 }}>{fruit.detail}</div>
    </>
  ) : (
    fruit.name
  );

describe('Picker', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const renderPickers = (
    count: number,
    {
      groups = FRUIT_GROUPS,
      isCompact,
      onSelect = vi.fn(),
      selectedId = null,
      status,
    }: {
      groups?: PickerGroup<Fruit>[];
      isCompact?: boolean;
      onSelect?: (fruit: Fruit) => void;
      selectedId?: string | null;
      status?: PickerStatus;
    } = {}
  ) =>
    act(() => {
      root.render(
        <ChakraProvider value={system}>
          {Array.from({ length: count }, (_, index) => (
            <Picker<Fruit>
              key={index}
              emptyMessage="No fruit installed"
              getIsOptionDisabled={getIsOptionDisabled}
              getOptionId={getOptionId}
              groups={groups}
              isCompact={isCompact}
              isMatch={isMatch}
              listLabel={`Fruit ${index}`}
              noMatchesMessage="No fruit matches"
              renderOption={renderOption}
              searchPlaceholder={`Search fruit ${index}`}
              selectedId={selectedId}
              status={status}
              onSelect={onSelect}
            />
          ))}
        </ChakraProvider>
      );
    });

  const searches = () => [...host.querySelectorAll<HTMLInputElement>('input[role="combobox"]')];
  const search = () => searches()[0]!;
  const activeOption = (input = search()) => {
    const id = input.getAttribute('aria-activedescendant');

    return id ? document.getElementById(id) : null;
  };
  const press = (key: string, init: KeyboardEventInit = {}) =>
    act(() => {
      search().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key, ...init }));
    });

  it('wires each search box to its own listbox and active option', async () => {
    await renderPickers(2, { selectedId: 'orange' });

    const listboxIds = searches().map((input) => {
      expect(input.getAttribute('aria-expanded')).toBe('true');
      expect(input.getAttribute('aria-autocomplete')).toBe('list');

      const listbox = document.getElementById(input.getAttribute('aria-controls')!)!;
      expect(listbox.getAttribute('role')).toBe('listbox');
      expect(listbox.getAttribute('aria-label')).toBe(
        input.getAttribute('aria-label')!.replace('Search fruit', 'Fruit')
      );

      // The selected option starts active, and the active option is one of this listbox's own options.
      const active = activeOption(input)!;
      expect(active.getAttribute('role')).toBe('option');
      expect(listbox.contains(active)).toBe(true);
      expect(active.textContent).toBe('Orange');
      expect(active.getAttribute('aria-selected')).toBe('true');

      return listbox.id;
    });

    expect(new Set(listboxIds).size).toBe(2);
    expect(new Set([...document.querySelectorAll('[id]')].map((element) => element.id)).size).toBe(
      document.querySelectorAll('[id]').length
    );

    const firstListbox = document.getElementById(listboxIds[0]!)!;
    const groups = [...firstListbox.querySelectorAll('[role="group"]')];
    expect(groups.map((group) => group.getAttribute('aria-label'))).toEqual(['Citrus', 'Berries']);
    expect(
      [...groups[0]!.querySelectorAll('[role="option"]')].map((option) => [
        option.textContent,
        option.getAttribute('aria-selected'),
        option.getAttribute('aria-disabled'),
        option.getAttribute('aria-posinset'),
        option.getAttribute('aria-setsize'),
      ])
    ).toEqual([
      ['Lemon', 'false', null, '1', '3'],
      ['Lime', 'false', 'true', '2', '3'],
      ['Orange', 'true', null, '3', '3'],
    ]);

    const violations = await auditAccessibility(host);
    expect(violations.filter((violation) => violation.id !== 'color-contrast')).toEqual([]);
    // Base-colored group labels on the panel, in both pickers.
    expect(contrastOffenderTexts(violations)).toEqual(['Berries', 'Berries', 'Citrus', 'Citrus']);
  });

  it('moves past disabled options with the arrows and selects the active one with Enter', async () => {
    const onSelect = vi.fn();
    await renderPickers(1, { onSelect });

    expect(activeOption()?.textContent).toBe('Lemon');
    await press('ArrowDown');
    expect(activeOption()?.textContent).toBe('Orange');
    await press('ArrowDown');
    expect(activeOption()?.textContent).toBe('Strawberry');
    await press('ArrowUp');
    await press('ArrowUp');
    expect(activeOption()?.textContent).toBe('Lemon');

    // A disabled option neither takes the pointer's highlight nor selects on click.
    const lime = [...host.querySelectorAll<HTMLElement>('[role="option"]')].find(
      (option) => option.textContent === 'Lime'
    )!;
    await userEvent.hover(lime);
    await act(() => lime.click());
    expect(activeOption()?.textContent).toBe('Lemon');
    expect(onSelect).not.toHaveBeenCalled();

    await press('Enter');
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(FRUIT_GROUPS[0]!.options[0]);
  });

  it('leaves arrows and Enter to an IME while it composes', async () => {
    const onSelect = vi.fn();
    await renderPickers(1, { onSelect });

    await press('ArrowDown', { isComposing: true });
    await press('Enter', { isComposing: true });
    // Safari's confirming keydown arrives after compositionend, marked only by keyCode 229.
    await press('Enter', { keyCode: 229 });

    expect(activeOption()?.textContent).toBe('Lemon');
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('announces loading, failure and empty searches outside the listbox', async () => {
    const onRetry = vi.fn();
    await renderPickers(1, { status: { message: 'Loading fruit' } });

    const statusText = () => host.querySelector('[role="status"]')?.textContent;
    expect(statusText()).toBe('Loading fruit');
    expect(host.querySelector('[role="listbox"]')).toBeNull();
    expect(search().getAttribute('aria-expanded')).toBe('false');
    expect(search().hasAttribute('aria-controls')).toBe(false);
    expect(search().hasAttribute('aria-activedescendant')).toBe(false);

    await renderPickers(1, { status: { isError: true, message: 'Fruit failed to load', onRetry } });
    expect(statusText()).toBe('Fruit failed to load');
    await act(() =>
      [...host.querySelectorAll('button')].find((button) => button.textContent === 'common.retry')!.click()
    );
    expect(onRetry).toHaveBeenCalledOnce();

    await renderPickers(1);
    expect(statusText()).toBe('');
    await act(() => userEvent.type(search(), 'kiwi'));
    await expect.poll(statusText).toBe('No fruit matches');
    expect(search().getAttribute('aria-expanded')).toBe('false');
  });

  const viewportOf = (element: Element) => element.closest<HTMLElement>('[data-part="viewport"]')!;
  /** Visible in the viewport and not under the pinned group header. */
  const isInView = (element: Element | null) => {
    if (!element) {
      return false;
    }

    const viewport = viewportOf(element).getBoundingClientRect();
    const pinned = viewportOf(element).querySelector('[aria-hidden="true"][style*="sticky"]');
    const top = pinned ? pinned.getBoundingClientRect().bottom : viewport.top;
    const row = element.getBoundingClientRect();

    return row.top >= top - 1 && row.bottom <= viewport.bottom + 1;
  };

  it('keeps a long list bounded while the active option stays rendered', async () => {
    await renderPickers(1, { groups: MANY_GROUPS });

    expect(host.querySelectorAll('[role="option"]').length).toBeLessThan(50);

    for (let step = 0; step < 120; step += 1) {
      await press('ArrowDown');
    }

    // The virtualizer settles scroll and measurement on later frames.
    await expect.poll(() => isInView(activeOption())).toBe(true);
    const active = activeOption()!;
    expect(active.textContent).toBe('Fruit 120');
    expect(active.getAttribute('aria-posinset')).toBe('121');
    expect(active.getAttribute('aria-setsize')).toBe('500');
    expect(host.querySelectorAll('[role="option"]').length).toBeLessThan(50);
  });

  it('opens on a deep selection in view and keeps the active option mounted while scrolled away', async () => {
    await renderPickers(1, { groups: DEEP_GROUPS, selectedId: 'plums-100' });

    // Rendered and named in the first commit, then settled in view under its group's pinned header.
    expect(activeOption()?.getAttribute('aria-selected')).toBe('true');
    await expect.poll(() => isInView(activeOption())).toBe(true);
    const active = activeOption()!;
    expect(active.textContent).toBe('Plum 100');
    expect(active.closest('[role="group"]')?.getAttribute('aria-label')).toBe('Plums');
    expect([active.getAttribute('aria-posinset'), active.getAttribute('aria-setsize')]).toEqual(['101', '150']);
    expect(viewportOf(active).querySelector('[style*="sticky"]')?.textContent).toContain('Plums');

    await press('ArrowDown');
    await expect.poll(() => activeOption()?.textContent).toContain('Plum 101');
    await expect.poll(() => isInView(activeOption())).toBe(true);

    // Scroll back to the top, repeating until the virtualizer has finished revealing the active option.
    const viewport = viewportOf(active);
    const scrollToTop = () => {
      viewport.scrollTop = 0;
      viewport.dispatchEvent(new Event('scroll'));

      return (
        viewport.scrollTop === 0 && host.querySelector('[role="option"]')?.textContent === 'Apple 0Apples detail 0'
      );
    };
    await expect.poll(() => act(scrollToTop)).toBe(true);
    // Far outside the rendered window, yet still the mounted target of aria-activedescendant.
    expect(activeOption()?.textContent).toContain('Plum 101');
    expect(isInView(activeOption())).toBe(false);
  });

  it('opens with the selected option fully in view wherever it sits, once rows are measured', async () => {
    // Every Sedum carries a description, so rows outgrow the estimates the opening offset starts from. With 28px
    // headers and 36px rows estimated, "sedum-6" is the last option the estimates place above the 288px fold.
    const groups: PickerGroup<Fruit>[] = ['Sedum', 'Fern', 'Moss'].map((name, groupIndex) => ({
      id: name,
      name,
      options: Array.from({ length: 9 }, (_, index) => ({
        detail: groupIndex === 0 || index % 2 === 1 ? `${name} detail ${index}` : undefined,
        id: `${name.toLowerCase()}-${index}`,
        name: `${name} ${index}`,
      })),
    }));
    const viewport = () => host.querySelector<HTMLElement>('[data-part="viewport"]')!;
    const nextFrame = () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      });
    // Settled: the scroll offset held across two frames.
    const settledInView = async () => {
      const before = viewport().scrollTop;
      await nextFrame();
      await nextFrame();

      return viewport().scrollTop === before && isInView(activeOption());
    };

    for (const isCompact of [false, true]) {
      for (const selectedId of ['sedum-0', 'sedum-6', 'fern-4', 'moss-7', 'moss-8']) {
        await act(() => root.render(null));
        await renderPickers(1, { groups, isCompact, selectedId });

        expect(activeOption()?.id.endsWith(selectedId)).toBe(true);
        await expect
          .poll(settledInView, { message: `${selectedId}${isCompact ? ', compact' : ''}`, timeout: 2000 })
          .toBe(true);
      }
    }
  });

  it('shows the first result after filtering a list scrolled far down', async () => {
    await renderPickers(1, { groups: DEEP_GROUPS, selectedId: 'grapes-140' });
    await expect.poll(() => isInView(activeOption())).toBe(true);

    await act(() => userEvent.type(search(), 'pear 1'));

    await expect.poll(() => activeOption()?.textContent).toContain('Pear 1');
    await expect.poll(() => isInView(activeOption())).toBe(true);
    expect(activeOption()?.getAttribute('aria-posinset')).toBe('1');
  });

  it('keeps its scroll and rows across content-equal re-renders, and reveals the active option for a new set', async () => {
    await renderPickers(1, { groups: DEEP_GROUPS, selectedId: 'plums-100' });
    await expect.poll(() => isInView(activeOption())).toBe(true);

    // Wheel away from the active option, repeating until the opening reveal has settled.
    const viewport = viewportOf(activeOption()!);
    const scrollTo = (offset: number) => () => {
      viewport.scrollTop = offset;
      viewport.dispatchEvent(new Event('scroll'));

      return viewport.scrollTop === offset && host.querySelector('[role="option"]')?.textContent?.startsWith('Apple');
    };
    await expect.poll(() => act(scrollTo(600))).toBe(true);
    const visibleRow = [...host.querySelectorAll<HTMLElement>('[role="option"]')].find(isInView)!;
    const visibleId = visibleRow.id;

    // Callers rebuild groups and options on unrelated renders; one option's data really changes.
    const rebuilt = DEEP_GROUPS.map((group) => ({
      ...group,
      options: group.options.map((option) =>
        visibleId.endsWith(encodeURIComponent(option.id))
          ? { ...option, name: `${option.name} renamed` }
          : { ...option }
      ),
    }));
    await renderPickers(1, { groups: rebuilt, selectedId: 'plums-100' });

    expect(viewport.scrollTop).toBe(600);
    expect(document.getElementById(visibleId)).toBe(visibleRow);
    expect(visibleRow.textContent).toContain('renamed');
    expect(isInView(activeOption())).toBe(false);

    // A different result set remounts the list on its active option.
    await renderPickers(1, { groups: rebuilt.slice(1), selectedId: 'plums-100' });
    await expect.poll(() => isInView(activeOption())).toBe(true);
    expect(activeOption()?.textContent).toContain('Plum 100');
  });
});
