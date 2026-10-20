import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { CopyIcon } from 'lucide-react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { Button, IconButton } from './Button';

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

const render = async (reason: string | undefined, handlers: { copyId: () => void; copyLog: () => void }) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <IconButton aria-label="Copy ID" tooltip="Copy ID" unavailableReason={reason} onClick={handlers.copyId}>
          <CopyIcon />
        </IconButton>
        <Button unavailableReason={reason} onClick={handlers.copyLog}>
          Copy log
        </Button>
      </ChakraProvider>
    );
    await Promise.resolve();
  });
};

const tooltipText = (): string | null | undefined => document.querySelector('[role="tooltip"]')?.textContent;

describe('unavailable buttons', () => {
  it('stay focusable, read as disabled and described by their reason, show it, and ignore activation', async () => {
    const handlers = { copyId: vi.fn(), copyLog: vi.fn() };
    await render(REASON, handlers);
    const copyId = page.getByRole('button', { name: 'Copy ID', exact: true });
    const copyLog = page.getByRole('button', { name: 'Copy log', exact: true });

    for (const button of [copyId, copyLog]) {
      await expect.element(button).toHaveAttribute('aria-disabled', 'true');
      await expect.element(button).toHaveAccessibleDescription(REASON);
    }
    // The hidden copy of the reason stays out of the name.
    await expect.element(copyLog).toHaveAccessibleName('Copy log');

    await act(() => userEvent.keyboard('{Tab}'));
    expect(document.activeElement).toBe(copyId.element());
    await vi.waitFor(() => expect(tooltipText()).toBe(`Copy ID${REASON}`));
    await act(() => userEvent.keyboard('{Enter}'));

    await act(() => userEvent.keyboard('{Tab}'));
    expect(document.activeElement).toBe(copyLog.element());
    await vi.waitFor(() => expect(tooltipText()).toBe(REASON));
    await act(() => userEvent.keyboard('{Enter}'));
    // Pointer tooling refuses aria-disabled targets; a click still reaches the element and must do nothing.
    await act(() => (copyLog.element() as HTMLButtonElement).click());

    expect(handlers.copyId).not.toHaveBeenCalled();
    expect(handlers.copyLog).not.toHaveBeenCalled();
  });

  it('act, with their own tooltip and no description, when available', async () => {
    const handlers = { copyId: vi.fn(), copyLog: vi.fn() };
    await render(undefined, handlers);
    const copyId = page.getByRole('button', { name: 'Copy ID', exact: true });

    await expect.element(copyId).not.toHaveAttribute('aria-disabled');
    await act(() => userEvent.hover(copyId));
    await vi.waitFor(() => expect(tooltipText()).toBe('Copy ID'));
    await act(() => userEvent.click(copyId));
    await act(() => userEvent.click(page.getByRole('button', { name: 'Copy log', exact: true })));

    expect(handlers.copyId).toHaveBeenCalledOnce();
    expect(handlers.copyLog).toHaveBeenCalledOnce();
  });
});
