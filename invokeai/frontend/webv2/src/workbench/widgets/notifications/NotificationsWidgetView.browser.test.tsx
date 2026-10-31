import type * as QueueModule from '@features/queue';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';
import type * as WidgetRegistryContextModule from '@workbench/WorkbenchWidgetRegistryContext';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { AppToaster, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import i18next from 'i18next';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

type Store = ReturnType<typeof createWorkbenchStore>;

const workbench = vi.hoisted(() => ({ store: null as Store | null }));
const retryItems = vi.hoisted(() => vi.fn<(itemIds: number[]) => Promise<number[]>>());
vi.mock('@features/queue', async (importOriginal) => ({
  ...(await importOriginal<typeof QueueModule>()),
  queueCommands: { retryItems },
}));
const store = () => workbench.store!;

// A real store without the provider's persistence; the view reads and acts through the same hooks.
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useOptionalWorkbenchCommands: () => workbench.store?.commands ?? null,
  useWorkbenchCommands: () => workbench.store!.commands,
  useWorkbenchQueries: () => workbench.store!.queries,
  useWorkbenchSelector: <Selected,>(selector: (snapshot: ReturnType<Store['getSnapshot']>) => Selected) =>
    useExternalStoreSelector(workbench.store!.subscribe, workbench.store!.getSnapshot, selector, shallowEqual),
}));
vi.mock('@workbench/WorkbenchWidgetRegistryContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WidgetRegistryContextModule>()),
  useWorkbenchWidgetRegistry: () => ({ getWidgetById, getWidgetsForRegion }),
}));

const { NotificationsWidgetView } = await import('./NotificationsWidgetView');

await i18next.use(initReactI18next).init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const popoverProps = { presentation: 'expanded', region: 'popover' } as WidgetViewProps;

const renderPopover = () =>
  act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <AppToaster />
        <NotificationsWidgetView {...popoverProps} />
      </ChakraProvider>
    )
  );

const button = (name: string) =>
  [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (candidate) => candidate.textContent === name || candidate.getAttribute('aria-label') === name
  );
const cards = () => [...(host?.querySelectorAll<HTMLElement>('li') ?? [])];

beforeEach(() => {
  workbench.store = createWorkbenchStore();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  // A failing test must not leave fake timers behind for the next one.
  vi.useRealTimers();
  await act(() => toaster.remove());
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  workbench.store = null;
});

describe('notifications widget', () => {
  it('shows each notice in the current language with its repeats, age and the buttons its toast offered', async () => {
    const failure = {
      actions: [
        { itemId: 41, kind: 'reveal-queue-item' },
        { itemIds: [41], kind: 'retry-queue-item' },
      ],
      kind: 'error',
      message: '“A castle at dusk”: CUDA out of memory',
      persistent: true,
      title: 'Run failed in Portraits',
      titleKey: 'notifications.runOutcome.failed',
      titleValues: { project: 'Portraits' },
    } as const;
    store().commands.notifications.add(failure);
    store().commands.notifications.add(failure);
    await renderPopover();

    const [card] = cards();
    expect(cards()).toHaveLength(1);
    expect(card!.textContent).toContain('Run failed in Portraits');
    expect(card!.textContent).toContain('×2');
    expect(card!.querySelector('time')?.textContent).toBe('now');
    expect([...card!.querySelectorAll('button')].map((candidate) => candidate.textContent)).toEqual(
      expect.arrayContaining(['Show in queue', 'Retry'])
    );
  });

  it('queues a retry once: the pressed Retry leaves the notice', async () => {
    retryItems.mockResolvedValue([99]);
    store().commands.notifications.add({
      actions: [{ itemIds: [71], kind: 'retry-queue-item' }],
      kind: 'error',
      title: 'Run failed',
    });
    await renderPopover();

    const retry = button('Retry')!;
    await act(() => userEvent.click(retry));
    await act(() => retry.click());

    await vi.waitFor(() =>
      expect(
        store()
          .getSnapshot()
          .notifications.map((notification) => notification.titleKey)
      ).toContain('widgets.queue.retryQueued')
    );
    expect(retryItems).toHaveBeenCalledOnce();
    // Only its dismiss is left on the failure notice.
    const failureCard = cards().find((card) => card.textContent?.includes('Run failed'))!;
    expect([...failureCard.querySelectorAll('button')].map((candidate) => candidate.textContent)).not.toContain(
      'Retry'
    );
  });

  it('dismisses from the keyboard, moving focus to the next notice, then the previous, then the list', async () => {
    store().commands.notifications.add({ kind: 'info', title: 'First' });
    store().commands.notifications.add({ kind: 'info', title: 'Second' });
    store().commands.notifications.add({ kind: 'info', title: 'Third' });
    await renderPopover();
    const dismissOf = (index: number) =>
      cards()[index]!.querySelector<HTMLButtonElement>('button[aria-label="Dismiss notification"]')!;
    const focusedCard = () => document.activeElement?.closest('li')?.textContent ?? '';
    const titles = () =>
      store()
        .getSnapshot()
        .notifications.map((notification) => notification.title);

    // Newest first: Third, Second, First.
    dismissOf(0).focus();
    await act(() => userEvent.keyboard('{Enter}'));
    expect(titles()).toEqual(['Second', 'First']);
    expect(focusedCard()).toContain('Second');

    dismissOf(1).focus();
    await act(() => userEvent.keyboard(' '));
    expect(titles()).toEqual(['Second']);
    expect(focusedCard()).toContain('Second');

    await act(() => userEvent.keyboard('{Enter}'));
    expect(titles()).toEqual([]);
    // With no notice left, focus rests on the list, visibly, not on the page.
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Notifications');
    expect(document.activeElement?.matches(':focus-visible')).toBe(true);
  });

  it('counts a notice as read once it has been on screen for a moment, even under StrictMode', async () => {
    store().commands.notifications.add({ kind: 'info', title: 'New' });
    await act(() =>
      root?.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <NotificationsWidgetView {...popoverProps} />
          </ChakraProvider>
        </StrictMode>
      )
    );

    // StrictMode's extra detach must not count as seen.
    expect(store().getSnapshot().notifications[0]?.isRead).toBe(false);
    await vi.waitFor(() => expect(store().getSnapshot().notifications[0]?.isRead).toBe(true), { timeout: 4_000 });
  });

  it('leaves notices scrolled out of view new while marking the ones on screen', async () => {
    for (let index = 0; index < 20; index += 1) {
      store().commands.notifications.add({ kind: 'info', message: `Notice ${index}`, title: `Notice ${index}` });
    }
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <Box display="flex" flexDirection="column" h="160px">
            <NotificationsWidgetView {...popoverProps} />
          </Box>
        </ChakraProvider>
      )
    );
    const isRead = (title: string) =>
      store()
        .getSnapshot()
        .notifications.find((notification) => notification.title === title)?.isRead;

    // Newest first: the top of the list is on screen, the oldest is far below it.
    await vi.waitFor(() => expect(isRead('Notice 19')).toBe(true), { timeout: 4_000 });
    expect(isRead('Notice 0')).toBe(false);
  });

  it('leaves a notice new when the list closes before it could be read', async () => {
    store().commands.notifications.add({ kind: 'info', title: 'New' });
    await renderPopover();
    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });

    await act(() => root?.render(<ChakraProvider value={system}>{null}</ChakraProvider>));
    await new Promise((resolve) => {
      setTimeout(resolve, 2_000);
    });

    expect(store().getSnapshot().notifications[0]?.isRead).toBe(false);
  });

  it('clears the list to empty, with an Undo toast that puts it back', async () => {
    store().commands.notifications.add({ kind: 'info', title: 'Kept' });
    await renderPopover();

    await act(() => userEvent.click(button('Clear')!));
    // The clear's own notice stays out of the list it emptied.
    expect(store().getSnapshot().notifications).toEqual([]);

    const undo = await vi.waitFor(() => {
      const candidate = [...document.querySelectorAll<HTMLButtonElement>('[data-scope="toast"] button')].find(
        (toastButton) => toastButton.textContent === 'Undo'
      );
      expect(candidate).toBeDefined();
      return candidate!;
    });
    await act(() => userEvent.click(undo));

    expect(
      store()
        .getSnapshot()
        .notifications.map((notification) => notification.title)
    ).toEqual(['Kept']);
  });
});
