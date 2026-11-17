import type * as QueueModule from '@features/queue';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';
import type * as WidgetRegistryContextModule from '@workbench/WorkbenchWidgetRegistryContext';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { AppToaster, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { createEditorNotificationActionContext } from '@workbench/shell/notifications/editorNotificationActions';
import { startNotificationToasts } from '@workbench/shell/notifications/notificationToasts';
import { registerNotificationUndo } from '@workbench/shell/notifications/undoRegistry';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import i18next from 'i18next';
import { act, StrictMode, type ReactNode } from 'react';
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
let stopToasts: (() => void) | null = null;
const popoverProps = { presentation: 'expanded', region: 'popover' } as WidgetViewProps;

/** The list loads with the popover, not with the chip; wait for it rather than for its loading fallback. */
const listLoaded = () =>
  vi.waitFor(() =>
    expect(
      host?.querySelector('[role="list"][aria-label="Notifications"], [role="group"][aria-label="No notifications"]')
    ).not.toBeNull()
  );

const renderView = async (view: ReactNode) => {
  await act(() => root?.render(<ChakraProvider value={system}>{view}</ChakraProvider>));
  await listLoaded();
};

const renderPopover = () =>
  renderView(
    <>
      <AppToaster />
      <NotificationsWidgetView {...popoverProps} />
    </>
  );

const button = (name: string) =>
  [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (candidate) => candidate.textContent === name || candidate.getAttribute('aria-label') === name
  );
const cards = () => [...(host?.querySelectorAll<HTMLElement>('[role="listitem"]') ?? [])];
const cardNamed = (title: string) => cards().find((card) => card.textContent?.includes(title));
const buttonsOf = (element: HTMLElement | undefined) =>
  [...(element?.querySelectorAll<HTMLButtonElement>('button') ?? [])].map((candidate) => candidate.textContent);
const toastNamed = (title: string) =>
  [...document.querySelectorAll<HTMLElement>('[data-scope="toast"][data-part="root"]')].find(
    (toast) => toast.querySelector('[data-part="title"]')?.textContent === title
  );

/** What `WorkbenchNotificationToaster` runs in the editor: toasts, and keeping undo entries in step with the list. */
const startToasts = () => {
  stopToasts = startNotificationToasts({
    actionContext: createEditorNotificationActionContext({
      focus: { focusFloating: () => undefined, focusRegion: () => undefined },
      getWidgetById,
      getWidgetsForRegion,
      notifications: store().commands.notifications,
      queries: store().queries,
      retryQueueItems: retryItems,
      t: i18next.t,
      widgets: store().commands.widgets,
    }),
    getNotifications: () => store().getSnapshot().notifications,
    getPreferences: () => ({ notifyOnEnqueue: true }),
    subscribe: store().subscribe,
  });
};

/** Records an undoable notice the way `useNotify().undoable` does inside the editor. */
const recordUndoable = (title: string, onUndo: () => void) =>
  store().commands.notifications.add({
    actions: [{ kind: 'undo', undoId: registerNotificationUndo(onUndo) }],
    kind: 'success',
    title,
  });

const buttonIn = (element: HTMLElement, name: string) =>
  [...element.querySelectorAll<HTMLButtonElement>('button')].find((candidate) => candidate.textContent === name)!;

/** Lets every open toast go, as its timeout would. */
const dismissToasts = async () => {
  await act(() => toaster.dismiss());
  await vi.waitFor(() => expect(document.querySelector('[data-scope="toast"][data-part="root"]')).toBeNull());
};

beforeEach(() => {
  workbench.store = createWorkbenchStore();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  // A failing test must not leave fake timers behind for the next one.
  vi.useRealTimers();
  stopToasts?.();
  stopToasts = null;
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
    const focusedCard = () => document.activeElement?.closest('[role="listitem"]')?.textContent ?? '';
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
    // With no notice left, focus rests on the empty state that says so, visibly, not on the page.
    await vi.waitFor(() => expect(document.activeElement?.getAttribute('aria-label')).toBe('No notifications'));
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
    await listLoaded();

    // StrictMode's extra detach must not count as seen.
    expect(store().getSnapshot().notifications[0]?.isRead).toBe(false);
    await vi.waitFor(() => expect(store().getSnapshot().notifications[0]?.isRead).toBe(true), { timeout: 4_000 });
  });

  it('leaves notices scrolled out of view new while marking the ones on screen', async () => {
    for (let index = 0; index < 20; index += 1) {
      store().commands.notifications.add({ kind: 'info', message: `Notice ${index}`, title: `Notice ${index}` });
    }
    await renderView(
      <Box display="flex" flexDirection="column" h="160px">
        <NotificationsWidgetView {...popoverProps} />
      </Box>
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

  it('keeps a notice’s Undo after its toast went, runs it once, and then drops it', async () => {
    const onUndo = vi.fn();
    startToasts();
    await renderPopover();
    await act(() => recordUndoable('Deleted preset', onUndo));
    await vi.waitFor(() => expect(buttonsOf(toastNamed('Deleted preset'))).toContain('Undo'));

    await dismissToasts();
    expect(buttonsOf(cardNamed('Deleted preset'))).toContain('Undo');
    const undo = buttonIn(cardNamed('Deleted preset')!, 'Undo');
    await act(() => userEvent.click(undo));
    await act(() => undo.click());

    expect(onUndo).toHaveBeenCalledOnce();
    expect(buttonsOf(cardNamed('Deleted preset'))).not.toContain('Undo');
  });

  it('closes the toast of an Undo pressed in the list', async () => {
    const onUndo = vi.fn();
    startToasts();
    await renderPopover();
    await act(() => recordUndoable('Deleted curve', onUndo));
    await vi.waitFor(() => expect(buttonsOf(toastNamed('Deleted curve'))).toContain('Undo'));
    // The first row of an empty list mounts once the list has measured its viewport.
    await vi.waitFor(() => expect(buttonsOf(cardNamed('Deleted curve'))).toContain('Undo'));

    await act(() => userEvent.click(buttonIn(cardNamed('Deleted curve')!, 'Undo')));

    expect(onUndo).toHaveBeenCalledOnce();
    expect(buttonsOf(cardNamed('Deleted curve'))).not.toContain('Undo');
    await vi.waitFor(() => expect(toastNamed('Deleted curve')?.getAttribute('data-state')).not.toBe('open'));
  });

  it('clears the list to empty, with an Undo toast that puts it back, Undo buttons included', async () => {
    const onUndo = vi.fn();
    startToasts();
    store().commands.notifications.add({ kind: 'info', title: 'Kept' });
    await renderPopover();
    await act(() => recordUndoable('Deleted preset', onUndo));
    // Only the list holds the notice's Undo now, so nothing but Clear's toast keeps it through the clear.
    await dismissToasts();

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
    ).toEqual(['Deleted preset', 'Kept']);
    // Toasts open a microtask after the record; the restored notices were seen once and stay quiet.
    await act(() => Promise.resolve());
    expect(toastNamed('Deleted preset')).toBeUndefined();
    expect(toastNamed('Kept')).toBeUndefined();
    await vi.waitFor(() => expect(buttonsOf(cardNamed('Deleted preset'))).toContain('Undo'));
    await act(() => userEvent.click(buttonIn(cardNamed('Deleted preset')!, 'Undo')));
    expect(onUndo).toHaveBeenCalledOnce();
  });

  it('names each row by its kind and unread state, which Mark read clears', async () => {
    store().commands.notifications.add({ kind: 'success', title: 'Saved' });
    store().commands.notifications.add({ kind: 'error', title: 'Upload failed' });
    // Checked before the rows have been on screen long enough to count as seen.
    await renderPopover();
    const rowName = (title: string) => cardNamed(title)?.querySelector('[data-list-primary]')?.textContent ?? '';

    // Kind is said in words and drawn as a distinct icon, never by colour alone.
    expect(rowName('Upload failed')).toMatch(/^Unread, Error: Upload failed/);
    expect(rowName('Saved')).toMatch(/^Unread, Success: Saved/);
    const iconOf = (title: string) => cardNamed(title)?.querySelector('svg')?.outerHTML;
    expect(iconOf('Upload failed')).not.toBe(iconOf('Saved'));

    await act(() => userEvent.click(button('Mark read')!));

    expect(
      store()
        .getSnapshot()
        .notifications.every((notification) => notification.isRead)
    ).toBe(true);
    expect(rowName('Upload failed')).toMatch(/^Error: Upload failed/);
    expect(button('Mark read')?.disabled).toBe(true);
  });

  it('shows the empty state once nothing is left, with the header actions disabled', async () => {
    await renderPopover();

    expect(host?.textContent).toContain('No notifications');
    expect(host?.textContent).toContain('Successful operations, errors, and system messages appear here.');
    expect(host?.querySelector('[role="list"]')).toBeNull();
    expect(button('Mark read')?.disabled).toBe(true);
    expect(button('Clear')?.disabled).toBe(true);
  });

  it('wraps a long message in full instead of clipping it', async () => {
    const message = `CUDA out of memory. ${'Tried to allocate 2.00 GiB on a card that is already nearly full. '.repeat(6)}`;
    store().commands.notifications.add({ kind: 'error', message, title: 'Run failed' });
    await renderView(
      <Box display="flex" flexDirection="column" maxH="28rem" w="26rem">
        <NotificationsWidgetView {...popoverProps} />
      </Box>
    );
    const card = cardNamed('Run failed')!;
    const text = [...card.querySelectorAll('span')].find((span) => span.textContent === message)!;

    expect(text).toBeDefined();
    // Several lines tall, inside its row, and no wider than the row.
    expect(text.getBoundingClientRect().height).toBeGreaterThan(40);
    expect(text.getBoundingClientRect().bottom).toBeLessThanOrEqual(card.getBoundingClientRect().bottom);
    expect(text.scrollWidth).toBeLessThanOrEqual(text.clientWidth);
  });

  it('scrolls a long list inside a height-capped popover, mounting only the rows near view', async () => {
    for (let index = 0; index < 60; index += 1) {
      store().commands.notifications.add({ kind: 'info', title: `Notice ${index}` });
    }
    await renderView(
      <Box display="flex" flexDirection="column" maxH="240px" w="26rem">
        <NotificationsWidgetView {...popoverProps} />
      </Box>
    );
    const viewport = host!.querySelector<HTMLElement>('[data-list-viewport]')!;

    expect(viewport.clientHeight).toBeLessThan(240);
    expect(viewport.scrollHeight).toBeGreaterThan(viewport.clientHeight);
    expect(cards().length).toBeLessThan(60);
  });

  it('moves between rows with the arrow keys and reaches a row’s actions with Tab', async () => {
    store().commands.notifications.add({
      actions: [{ itemIds: [7], kind: 'retry-queue-item' }],
      kind: 'error',
      title: 'First',
    });
    store().commands.notifications.add({ kind: 'info', title: 'Second' });
    store().commands.notifications.add({ kind: 'info', title: 'Third' });
    await renderPopover();
    const primaryOf = (title: string) => cardNamed(title)!.querySelector<HTMLElement>('[data-list-primary]')!;
    const focusedRow = () => document.activeElement?.closest('[role="listitem"]')?.textContent ?? '';

    // One tab stop for the rows: the newest.
    expect(
      [...host!.querySelectorAll<HTMLElement>('[data-list-primary]')].filter((primary) => primary.tabIndex === 0)
    ).toEqual([primaryOf('Third')]);

    primaryOf('Third').focus();
    await act(() => userEvent.keyboard('{ArrowDown}'));
    expect(document.activeElement).toBe(primaryOf('Second'));
    await act(() => userEvent.keyboard('{End}'));
    expect(document.activeElement).toBe(primaryOf('First'));

    await act(() => userEvent.keyboard('{Tab}'));
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Dismiss notification');
    expect(focusedRow()).toContain('First');
    await act(() => userEvent.keyboard('{Tab}'));
    expect(document.activeElement?.textContent).toBe('Retry');

    // Arrows from a row's action move on to the neighbouring row.
    await act(() => userEvent.keyboard('{ArrowUp}'));
    expect(document.activeElement).toBe(primaryOf('Second'));
    await act(() => userEvent.keyboard('{Home}'));
    expect(document.activeElement).toBe(primaryOf('Third'));
  });

  it('opens a row’s menu from its button, offering the notice’s actions and Dismiss', async () => {
    store().commands.notifications.add({ kind: 'info', title: 'Older' });
    store().commands.notifications.add({
      actions: [
        { itemId: 41, kind: 'reveal-queue-item' },
        { itemIds: [41], kind: 'retry-queue-item' },
      ],
      kind: 'error',
      title: 'Run failed',
    });
    await renderPopover();
    const primary = cardNamed('Run failed')!.querySelector<HTMLElement>('[data-list-primary]')!;
    const menuItems = () =>
      [...document.querySelectorAll<HTMLElement>('[data-scope="menu"][data-part="item"]')].map(
        (item) => item.textContent
      );

    // The row's button says it opens a menu, and pressing it does.
    expect(primary.getAttribute('aria-haspopup')).toBe('menu');
    primary.focus();
    await act(() => userEvent.keyboard('{Enter}'));
    await vi.waitFor(() => expect(menuItems()).toEqual(['Show in queue', 'Retry', 'Dismiss notification']));
    expect(primary.getAttribute('aria-expanded')).toBe('true');

    const dismiss = [...document.querySelectorAll<HTMLElement>('[data-scope="menu"][data-part="item"]')].find(
      (item) => item.textContent === 'Dismiss notification'
    )!;
    await act(() => userEvent.click(dismiss));

    expect(
      store()
        .getSnapshot()
        .notifications.map((notification) => notification.title)
    ).toEqual(['Older']);
    // As with the inline Dismiss, focus moves on to the next notice's dismiss.
    await vi.waitFor(() => expect(document.activeElement?.getAttribute('aria-label')).toBe('Dismiss notification'));
    expect(document.activeElement?.closest('[role="listitem"]')?.textContent).toContain('Older');
  });
});
