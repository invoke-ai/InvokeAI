import { ChakraProvider, Menu } from '@chakra-ui/react';
import { system } from '@theme/system';
import { CopyIcon } from 'lucide-react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { MenuActionItem, MenuIconItem, MenuItemReason, useUnavailableMenuItem } from './Menu';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const REASON = 'Copying needs HTTPS or localhost';

/** A plain `Menu.Item`, as feature menus write theirs. */
const PlainUnavailableItem = ({ onClick, reason }: { onClick: () => void; reason: string | undefined }) => {
  const unavailable = useUnavailableMenuItem(reason);
  return (
    <Menu.Item {...unavailable.itemProps} value="copy-path" onClick={reason ? undefined : onClick}>
      <MenuItemReason reason={reason}>Copy path</MenuItemReason>
      {unavailable.description}
    </Menu.Item>
  );
};

const handlers = () => ({ copyImage: vi.fn(), copyJson: vi.fn(), copyPath: vi.fn(), rename: vi.fn() });

const renderMenu = async (actions: ReturnType<typeof handlers>) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <Menu.Root>
          <Menu.Trigger>Actions</Menu.Trigger>
          <Menu.Positioner>
            <Menu.Content>
              <MenuIconItem
                icon={CopyIcon}
                label="Copy image"
                unavailableReason={REASON}
                value="copy-image"
                onSelect={actions.copyImage}
              />
              <MenuActionItem label="Rename" value="rename" onSelect={actions.rename} />
              <MenuActionItem
                label="Copy JSON"
                unavailableReason={REASON}
                value="copy-json"
                onSelect={actions.copyJson}
              />
              <PlainUnavailableItem reason={REASON} onClick={actions.copyPath} />
            </Menu.Content>
          </Menu.Positioner>
        </Menu.Root>
      </ChakraProvider>
    );
    await Promise.resolve();
  });
};

const item = (value: string): HTMLElement => document.querySelector<HTMLElement>(`[data-value="${value}"]`)!;
const tooltipText = (): string | null | undefined => document.querySelector('[role="tooltip"]')?.textContent;

const highlight = async (value: string) => {
  await act(() => userEvent.keyboard('{ArrowDown}'));
  await vi.waitFor(() => expect(item(value).hasAttribute('data-highlighted')).toBe(true));
};

describe('unavailable menu items', () => {
  it('stay reachable by keyboard, read as disabled and described by their reason, and never act', async () => {
    const actions = handlers();
    await renderMenu(actions);

    await act(() => userEvent.click(page.getByRole('button', { name: 'Actions' })));
    await vi.waitFor(() => expect(item('rename')).toBeVisible());

    // Their name stays the label; the reason is their description, read without the tooltip.
    for (const [value, name] of [
      ['copy-image', 'Copy image'],
      ['copy-json', 'Copy JSON'],
      ['copy-path', 'Copy path'],
    ] as const) {
      expect(item(value).getAttribute('aria-disabled')).toBe('true');
      const menuItem = page.getByRole('menuitem', { name, exact: true });
      await expect.element(menuItem).toHaveAccessibleName(name);
      await expect.element(menuItem).toHaveAccessibleDescription(REASON);
    }

    // Keyboard navigation does not skip them, and highlighting shows the reason without a pointer. The icon-only
    // item shows its label beside the reason, since the tooltip is the only place its name appears.
    await highlight('copy-image');
    await vi.waitFor(() => expect(tooltipText()).toBe(`Copy image${REASON}`));
    await act(() => userEvent.keyboard('{Enter}'));

    await highlight('rename');
    await highlight('copy-json');
    await vi.waitFor(() => expect(tooltipText()).toBe(REASON));
    await act(() => userEvent.keyboard('{Enter}'));

    await highlight('copy-path');
    await act(() => userEvent.keyboard('{Enter}'));

    expect(actions.copyImage).not.toHaveBeenCalled();
    expect(actions.copyJson).not.toHaveBeenCalled();
    expect(actions.copyPath).not.toHaveBeenCalled();
    // Selecting an unavailable item leaves the menu open, as a disabled one would.
    expect(item('rename')).toBeVisible();
  });
});
