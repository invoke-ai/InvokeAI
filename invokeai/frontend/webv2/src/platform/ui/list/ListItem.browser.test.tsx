/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ListItem } from './ListItem';

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
});
