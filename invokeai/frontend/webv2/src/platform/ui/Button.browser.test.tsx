import type { ReactNode } from 'react';

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
  await userEvent.unhover(document.body);
  await act(() => root?.unmount());
  host?.remove();
  delete document.documentElement.dataset.reduceMotion;
  host = null;
  root = null;
});

const REASON = 'Copying needs HTTPS or localhost';

const mount = async (children: ReactNode) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(async () => {
    root?.render(<ChakraProvider value={system}>{children}</ChakraProvider>);
    await Promise.resolve();
  });
};

const render = (reason: string | undefined, handlers: { copyId: () => void; copyLog: () => void }) =>
  mount(
    <>
      <IconButton aria-label="Copy ID" tooltip="Copy ID" unavailableReason={reason} onClick={handlers.copyId}>
        <CopyIcon />
      </IconButton>
      <Button unavailableReason={reason} onClick={handlers.copyLog}>
        Copy log
      </Button>
    </>
  );

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

  it('do not submit their form, by click, key, or implicit submission', async () => {
    const submit = vi.fn((event: SubmitEvent) => event.preventDefault());
    await mount(
      <form onSubmit={(event) => submit(event.nativeEvent as SubmitEvent)}>
        <input aria-label="Name" />
        <Button type="submit" unavailableReason={REASON}>
          Save
        </Button>
      </form>
    );
    const ancestorClick = vi.fn();
    host?.addEventListener('click', ancestorClick);
    const save = page.getByRole('button', { name: 'Save', exact: true });

    await act(() => (save.element() as HTMLButtonElement).click());
    (save.element() as HTMLButtonElement).focus();
    await act(() => userEvent.keyboard('{Enter}'));
    await act(() => userEvent.keyboard(' '));
    // Enter in a field submits through the form's default button, which is this one.
    (page.getByRole('textbox', { name: 'Name' }).element() as HTMLInputElement).focus();
    await act(() => userEvent.keyboard('{Enter}'));

    expect(submit).not.toHaveBeenCalled();
    // Like a natively disabled button, it delivers no click to its ancestors either.
    expect(ancestorClick).not.toHaveBeenCalled();
  });

  it('keep the description the caller gives, beside the reason', async () => {
    await mount(
      <>
        <p id="copy-hint">Copies the whole log</p>
        <Button aria-describedby="copy-hint" unavailableReason={REASON}>
          Copy log
        </Button>
        <IconButton aria-describedby="copy-hint" aria-label="Copy ID" unavailableReason={REASON}>
          <CopyIcon />
        </IconButton>
      </>
    );

    for (const name of ['Copy log', 'Copy ID']) {
      await expect
        .element(page.getByRole('button', { name, exact: true }))
        .toHaveAccessibleDescription(`Copies the whole log ${REASON}`);
    }
  });

  it('keep their reason reachable when the caller also disables them', async () => {
    const copyLog = vi.fn();
    await mount(
      <Button disabled unavailableReason={REASON} onClick={copyLog}>
        Copy log
      </Button>
    );
    const button = page.getByRole('button', { name: 'Copy log', exact: true });

    await expect.element(button).not.toHaveAttribute('disabled');
    await expect.element(button).toHaveAttribute('aria-disabled', 'true');
    await act(() => userEvent.keyboard('{Tab}'));
    expect(document.activeElement).toBe(button.element());
    await vi.waitFor(() => expect(tooltipText()).toBe(REASON));
    await act(() => userEvent.keyboard('{Enter}'));
    expect(copyLog).not.toHaveBeenCalled();
  });

  it('show no hover feedback, even while their reason shows', async () => {
    // Read settled colors: reduced motion shortens the hover transition to a millisecond.
    document.documentElement.dataset.reduceMotion = 'true';
    await mount(
      <>
        <Button variant="outline">Available</Button>
        <Button unavailableReason={REASON} variant="outline">
          Unavailable
        </Button>
      </>
    );
    const background = (name: string) =>
      getComputedStyle(page.getByRole('button', { name, exact: true }).element()).backgroundColor;
    const resting = { available: background('Available'), unavailable: background('Unavailable') };

    // The control: hovering an available button does change its fill.
    await act(() => userEvent.hover(page.getByRole('button', { name: 'Available', exact: true })));
    await vi.waitFor(() => expect(background('Available')).not.toBe(resting.available));

    await act(() => userEvent.hover(page.getByRole('button', { name: 'Unavailable', exact: true })));
    await vi.waitFor(() => expect(background('Available')).toBe(resting.available));
    expect(background('Unavailable')).toBe(resting.unavailable);
    // The open tooltip marks its trigger expanded, which buttons tint like hover.
    await vi.waitFor(() => expect(tooltipText()).toBe(REASON));
    expect(background('Unavailable')).toBe(resting.unavailable);
  });
});
