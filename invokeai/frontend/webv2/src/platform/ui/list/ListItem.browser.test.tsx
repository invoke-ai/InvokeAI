/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider, Field, Input } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ListItem } from './ListItem';
import { ListSelectionBar } from './ListSelectionBar';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'width:320px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const render = async (element: ReactElement) => {
  await act(() => root.render(<ChakraProvider value={system}>{element}</ChakraProvider>));
};

const primary = () => host.querySelector<HTMLElement>('[data-list-primary]')!;

describe('ListItem', () => {
  it('keeps actions beside the primary button and announces disclosure', async () => {
    const onPress = vi.fn();
    const onAction = vi.fn();

    await render(
      <ListItem
        actions={
          <button type="button" onClick={onAction}>
            Cancel
          </button>
        }
        isExpanded={false}
        title="A prompt"
        onPress={onPress}
      />
    );

    const action = host.querySelector<HTMLButtonElement>('button:not([data-list-primary])')!;

    expect(primary().tagName).toBe('BUTTON');
    expect(primary().getAttribute('aria-expanded')).toBe('false');
    expect(primary().contains(action)).toBe(false);
    expect(action.closest('[role="listitem"]')).not.toBeNull();

    await act(() => action.click());
    expect(onAction).toHaveBeenCalledTimes(1);
    expect(onPress).not.toHaveBeenCalled();
    // Leave a real pointer over the row: the next test must not inherit a hover on its new controls.
    await userEvent.hover(primary());
  });

  it('keeps detail on the row surface outside the primary button, deferring to a menu it handles itself', async () => {
    const onContextMenu = vi.fn();

    await render(
      <ListItem
        detail={
          <div>
            <input aria-label="Weight" onContextMenu={(event) => event.preventDefault()} />
            <span>Plain detail</span>
          </div>
        }
        title="A concept"
        onContextMenu={onContextMenu}
      />
    );

    const input = host.querySelector('input')!;

    expect(input.closest('[role="listitem"]')).not.toBeNull();
    expect(primary().contains(input)).toBe(false);

    input.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    expect(onContextMenu).not.toHaveBeenCalled();

    host.querySelector('span')!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    expect(onContextMenu).toHaveBeenCalledTimes(1);
  });

  it('keeps the hover surface while its context menu is open', async () => {
    const surface = () => getComputedStyle(primary().parentElement!).backgroundColor;

    await userEvent.unhover(document.body);
    await render(<ListItem title="A model" onContextMenu={vi.fn()} />);
    const resting = surface();

    await render(<ListItem isMenuOpen title="A model" onContextMenu={vi.fn()} />);

    expect(resting).toBe('rgba(0, 0, 0, 0)');
    expect(surface()).not.toBe(resting);
  });

  it('keeps a disabled row focusable but inert: no press, checkbox, menu, intent, actions, or hover fill', async () => {
    const onAction = vi.fn();
    const onCheckedChange = vi.fn();
    const onContextMenu = vi.fn();
    const onIntent = vi.fn();
    const onPress = vi.fn();
    const surface = () => getComputedStyle(primary().parentElement!).backgroundColor;

    // Transitions finish within a frame, so a hover fill would already show when the background is read.
    document.documentElement.dataset.reduceMotion = 'true';
    try {
      await userEvent.unhover(document.body);
      await render(
        <ListItem
          actions={
            <button type="button" onClick={onAction}>
              Install
            </button>
          }
          isDisabled
          title="Too new"
          onCheckedChange={onCheckedChange}
          onContextMenu={onContextMenu}
          onIntent={onIntent}
          onPress={onPress}
        />
      );
      const resting = surface();

      expect(primary().getAttribute('aria-disabled')).toBe('true');
      expect(primary().hasAttribute('aria-haspopup')).toBe(false);
      await act(() => primary().focus());
      expect(document.activeElement).toBe(primary());

      // Force past Playwright's actionability checks: disabled and inert controls are exactly what is under test.
      await userEvent.hover(primary(), { force: true });
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      });
      expect(surface()).toBe(resting);
      await userEvent.click(primary(), { force: true });
      await userEvent.keyboard('{Enter}');
      await act(() => primary().dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true })));
      await userEvent.click(host.querySelector<HTMLElement>('[data-part="control"]')!, { force: true });
      await userEvent.click(
        [...host.querySelectorAll('button')].find((button) => button.textContent === 'Install')!,
        { force: true }
      );

      expect(onPress).not.toHaveBeenCalled();
      expect(onCheckedChange).not.toHaveBeenCalled();
      expect(onContextMenu).not.toHaveBeenCalled();
      expect(onIntent).not.toHaveBeenCalled();
      expect(onAction).not.toHaveBeenCalled();
    } finally {
      delete document.documentElement.dataset.reduceMotion;
    }
  });

  it('dims a disabled row’s content but draws its keyboard focus ring at full strength', async () => {
    await render(<ListItem isDisabled title="Too new" onCheckedChange={vi.fn()} onPress={vi.fn()} />);
    const row = primary().parentElement!;

    // Tab past the disabled checkbox onto the row button, so the ring is keyboard focus-visible.
    await act(() => (document.activeElement as HTMLElement | null)?.blur());
    await userEvent.keyboard('{Tab}');
    expect(document.activeElement).toBe(primary());

    expect(getComputedStyle(row).outlineStyle).toBe('solid');
    expect(getComputedStyle(row).outlineWidth).toBe('2px');
    expect(getComputedStyle(row).opacity).toBe('1');
    // Checkbox slot and primary button: everything the row shows.
    expect([...row.children].map((slot) => getComputedStyle(slot).opacity)).toEqual(['0.5', '0.5']);
  });

  it('renders a static row without a press handler as plain content, not a button', async () => {
    await render(<ListItem title="Read only" />);

    expect(primary().tagName).toBe('DIV');
    expect(primary().hasAttribute('tabindex')).toBe(false);
    expect(host.querySelector('button')).toBeNull();
  });

  it('warms on hover or focus before the press and truncates prose at the end', async () => {
    const onIntent = vi.fn();
    const longPrompt = 'a very long prompt that keeps going past the width of this narrow row in the harness';

    await render(<ListItem title={longPrompt} titleTruncate="end" onIntent={onIntent} onPress={() => undefined} />);

    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    });
    expect(onIntent).not.toHaveBeenCalled();
    await act(() => primary().focus());
    expect(onIntent).toHaveBeenCalledTimes(1);
    await userEvent.hover(primary());
    expect(onIntent).toHaveBeenCalledTimes(2);

    // With end truncation there is no pinned tail: the title is one clipped run holding the whole text.
    const title = host.querySelector<HTMLElement>('[data-list-primary] [title]')!;

    expect(title.children).toHaveLength(1);
    expect(title.children[0]!.textContent).toBe(longPrompt);

    await render(<ListItem title={longPrompt} onIntent={onIntent} onPress={() => undefined} />);
    expect(host.querySelector<HTMLElement>('[data-list-primary] [title]')!.children).toHaveLength(2);
  });

  it('toggles the checkbox from the padding around it without stealing presses from the primary button', async () => {
    const onCheckedChange = vi.fn();
    const onPress = vi.fn();
    const item = (isChecked: boolean) => (
      <ListItem isChecked={isChecked} title="A model" onCheckedChange={onCheckedChange} onPress={onPress} />
    );

    await render(item(false));

    const row = primary().parentElement!;
    const rowRect = row.getBoundingClientRect();
    const control = host.querySelector<HTMLElement>('[data-part="control"]')!.getBoundingClientRect();
    const middle = control.top + control.height / 2 - rowRect.top;

    // The row's start padding and the space above the control, both outside its visible box.
    await userEvent.click(row, { position: { x: control.left - rowRect.left - 4, y: middle } });
    expect(onCheckedChange).toHaveBeenLastCalledWith(true);
    await render(item(true));
    await userEvent.click(row, { position: { x: control.left - rowRect.left + 2, y: control.top - rowRect.top - 4 } });
    expect(onCheckedChange).toHaveBeenLastCalledWith(false);
    expect(onPress).not.toHaveBeenCalled();

    // Past the outset, the primary button's start padding still belongs to the press.
    await userEvent.click(row, { position: { x: control.right - rowRect.left + 8, y: middle } });
    expect(onPress).toHaveBeenCalledTimes(1);
    expect(onCheckedChange).toHaveBeenCalledTimes(2);
  });

  it('keeps a title with line breaks on one line', async () => {
    await render(<ListItem title="single line" titleTruncate="end" onPress={() => undefined} />);
    const singleLineHeight = host
      .querySelector<HTMLElement>('[data-list-primary] [title]')!
      .getBoundingClientRect().height;

    await render(
      <ListItem title={'first line\nsecond line\nthird line'} titleTruncate="end" onPress={() => undefined} />
    );
    const title = host.querySelector<HTMLElement>('[data-list-primary] [title]')!;

    expect(title.getBoundingClientRect().height).toBe(singleLineHeight);
    expect(title.textContent).toContain('third line');
  });

  it('edits the title in place outside any button, at the same height, without the row menu', async () => {
    const onContextMenu = vi.fn();
    const row = (titleEditor?: ReactElement) => (
      <ListItem
        actions={<button type="button">Toggle</button>}
        density="snug"
        description="Summary"
        leading={<span style={{ display: 'block', height: 36, width: 36 }} />}
        title="Hero pose"
        titleEditor={titleEditor}
        onContextMenu={onContextMenu}
      />
    );

    await render(row());
    const surface = host.querySelector<HTMLElement>('[role="listitem"]')!;
    const restingHeight = surface.getBoundingClientRect().height;

    expect(primary().tagName).toBe('BUTTON');
    expect(primary().getAttribute('aria-haspopup')).toBe('menu');

    await render(row(<input aria-label="Rename" defaultValue="Hero pose" style={{ height: 20, minWidth: 0 }} />));
    const field = host.querySelector<HTMLInputElement>('input[aria-label="Rename"]')!;

    // The field takes the title's place inside the primary area, which no longer announces itself as a button.
    expect(primary().tagName).toBe('DIV');
    expect(primary().hasAttribute('tabindex')).toBe(false);
    expect(primary().hasAttribute('aria-haspopup')).toBe(false);
    expect(primary().contains(field)).toBe(true);
    expect(field.closest('button')).toBeNull();
    expect(host.textContent).not.toContain('Hero pose');
    expect(host.textContent).toContain('Summary');
    expect(host.querySelector('[role="listitem"]')).toBe(surface);
    expect(surface.getBoundingClientRect().height).toBe(restingHeight);

    // A right-click in the field keeps the browser's text menu instead of opening the row's.
    const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 10, clientY: 10 });
    field.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(onContextMenu).not.toHaveBeenCalled();

    // Done editing, the title and its button come back.
    await render(row());
    expect(primary().tagName).toBe('BUTTON');
    expect(host.textContent).toContain('Hero pose');
  });
});

describe('list checkboxes inside a form Field', () => {
  // A popover opened from a field's label keeps the Field's context; its list must not bind to the Field's input.
  it('toggle their own rows on a click and leave the Field input alone', async () => {
    const onSelectAll = vi.fn();
    const onRow = vi.fn();

    await render(
      <Field.Root>
        <Input aria-label="Field input" />
        <ListSelectionBar checked={false} label="Select all" onCheckedChange={onSelectAll} />
        <ListItem title="first" onCheckedChange={onRow} />
        <ListItem title="second" onCheckedChange={vi.fn()} />
      </Field.Root>
    );
    const input = host.querySelector<HTMLInputElement>('input[aria-label="Field input"]')!;
    const controls = host.querySelectorAll<HTMLElement>('[data-scope="checkbox"][data-part="control"]');
    const ids = [...host.querySelectorAll('input')].map((element) => element.id);

    expect(new Set(ids).size).toBe(ids.length);
    await act(() => userEvent.click(controls[0]!));
    await act(() => userEvent.click(controls[1]!));

    expect(onSelectAll).toHaveBeenCalledOnce();
    expect(onRow).toHaveBeenCalledOnce();
    expect(document.activeElement).not.toBe(input);
  });
});
