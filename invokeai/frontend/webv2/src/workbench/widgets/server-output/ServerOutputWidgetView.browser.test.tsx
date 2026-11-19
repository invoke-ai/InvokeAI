import type * as IdentityModule from '@features/identity';
import type * as SettingsStore from '@workbench/settings/store';
import type { WidgetRuntimeApi, WidgetViewProps } from '@workbench/widgetContracts';

import { Box, ChakraProvider, HStack } from '@chakra-ui/react';
import { queryClient } from '@platform/query/client';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { refetchFrontendConfig } from '@workbench/frontendConfig';
import { useWidgetCapabilities } from '@workbench/widgetCapabilities';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type * as ReaderModule from './reader';
import type { ServerOutputRead, ServerOutputRow } from './transcript';

import { serverOutputWidgetManifest } from './manifest';
import { ServerOutputHeaderActions, ServerOutputHeaderLabel } from './ServerOutputHeader';
import { ServerOutputWidgetView } from './ServerOutputWidgetView';

type Session = IdentityModule.AuthSession;

const auth = vi.hoisted(() => ({ listeners: new Set<() => void>(), session: null as unknown as Session }));
const preferences = vi.hoisted(() => ({ listeners: new Set<() => void>(), wrapLines: true }));

vi.mock('@features/identity', async (importOriginal) => {
  const { useSyncExternalStore } = await import('react');
  const subscribe = (listener: () => void) => {
    auth.listeners.add(listener);
    return () => auth.listeners.delete(listener);
  };

  return {
    ...(await importOriginal<typeof IdentityModule>()),
    getAuthSession: () => auth.session,
    subscribeAuthSession: subscribe,
    useAuthSession: () => useSyncExternalStore(subscribe, () => auth.session),
  };
});

// The account's preferences stand in for the settings store, which would hydrate from and write to the server.
vi.mock('@workbench/settings/store', async (importOriginal) => {
  const { useSyncExternalStore } = await import('react');
  const original = await importOriginal<typeof SettingsStore>();
  const subscribe = (listener: () => void) => {
    preferences.listeners.add(listener);
    return () => preferences.listeners.delete(listener);
  };

  return {
    ...original,
    useWorkbenchPreferenceSelector: <Selected,>(
      selector: (value: typeof original.DEFAULT_PREFERENCES) => Selected
    ): Selected =>
      selector({
        ...original.DEFAULT_PREFERENCES,
        serverOutputWrapLines: useSyncExternalStore(subscribe, () => preferences.wrapLines),
      }),
  };
});

// Real scheduling at a fraction of the production delays, so tests wait milliseconds rather than seconds.
vi.mock('./reader', async (importOriginal) => ({
  ...(await importOriginal<typeof ReaderModule>()),
  SERVER_OUTPUT_READER_TIMING: { activeMs: 15, errorMaxMs: 60, errorMs: 20, idleMaxMs: 40, idleMs: 20 },
}));

const singleUser: Session = {
  accountEpoch: 1,
  expiredUserId: null,
  multiuserEnabled: false,
  phase: 'ready',
  sessionExpired: false,
  setupRequired: false,
  strictPasswordChecking: false,
  switchingAccount: false,
  user: null,
};

const multiuser = (isAdmin: boolean): Session => ({
  ...singleUser,
  multiuserEnabled: true,
  user: {
    created_at: '',
    display_name: null,
    email: 'ada@example.com',
    is_active: true,
    is_admin: isAdmin,
    last_login_at: null,
    updated_at: '',
    user_id: 'ada',
  },
});

const setSession = (session: Session) => {
  auth.session = session;
  for (const listener of auth.listeners) {
    listener();
  }
};

const setWrapLines = (wrapLines: boolean) => {
  preferences.wrapLines = wrapLines;
  for (const listener of preferences.listeners) {
    listener();
  }
};

/** Follows the read contract with a cursor of `<run>:<revision>`; tests script output and failures on top. */
const createServer = () => {
  let run = 1;
  let revision = 0;
  let nextId = 1;
  let floor = 1;
  let expireNext = false;
  const rows = new Map<number, { row: ServerOutputRow; revision: number }>();
  const write = (row: ServerOutputRow) => {
    revision += 1;
    rows.set(row.id, { revision, row });
  };
  const snapshot = (reset: ServerOutputRead['reset']): ServerOutputRead => ({
    cursor: `${run}:${revision}`,
    floor_id: floor,
    has_more: false,
    reset,
    rows: [...rows.values()].map((entry) => entry.row).sort((left, right) => left.id - right.id),
    run_id: `run-${run}`,
    snapshot_truncated: floor > 1,
  });

  return {
    append: (...lines: Array<string | Pick<ServerOutputRow, 'styles' | 'text'>>) => {
      for (const line of lines) {
        write({ id: nextId, truncated: false, ...(typeof line === 'string' ? { styles: [], text: line } : line) });
        nextId += 1;
      }
    },
    evictBelow: (id: number) => {
      floor = id;
      for (const key of rows.keys()) {
        if (key < id) {
          rows.delete(key);
        }
      }
    },
    expire: () => {
      expireNext = true;
    },
    read: (cursor: string | null): ServerOutputRead => {
      if (cursor === null) {
        return snapshot('initial');
      }
      const [cursorRun, cursorRevision] = cursor.split(':').map(Number);
      if (cursorRun !== run) {
        return snapshot('restarted');
      }
      if (expireNext) {
        expireNext = false;
        return snapshot('expired');
      }
      return {
        ...snapshot(null),
        rows: [...rows.values()]
          .filter((entry) => entry.revision > cursorRevision!)
          .map((entry) => entry.row)
          .sort((left, right) => left.id - right.id),
      };
    },
    restart: () => {
      run += 1;
      revision = 0;
      nextId = 1;
      floor = 1;
      rows.clear();
    },
    rewrite: (id: number, text: string) => write({ id, styles: [], text, truncated: false }),
  };
};

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' }, status });

let server: ReturnType<typeof createServer>;
let config: { can_view_server_output?: boolean; show_donation_link: boolean } | 'fail';
let configReads: number;
let outputReads: Array<string | null>;
/** Answers a transcript read in place of the server; return undefined to let the server answer. */
let intercept: ((cursor: string | null) => Promise<Response> | undefined) | null;
let host: HTMLDivElement;
let root: Root;
const realFetch = globalThis.fetch;
const closeWidgetInstance = vi.fn(() => ({ ok: true as const }));
const queryDefaults = queryClient.getDefaultOptions();

beforeEach(() => {
  accountLifecycle.activate('server-output-test');
  // A failed config read retries once after a second in the app; tests fail it once.
  queryClient.setDefaultOptions({ ...queryDefaults, queries: { ...queryDefaults.queries, retry: false } });
  setSession(singleUser);
  setWrapLines(true);
  server = createServer();
  config = { can_view_server_output: true, show_donation_link: false };
  configReads = 0;
  outputReads = [];
  intercept = null;
  closeWidgetInstance.mockClear();
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const url = new URL(input instanceof Request ? input.url : String(input), window.location.href);
    if (url.pathname.endsWith('/api/v1/app/frontend_config')) {
      configReads += 1;
      return config === 'fail' ? json(500, { detail: 'Unavailable' }) : json(200, config);
    }
    if (url.pathname.endsWith('/api/v1/app/server_output')) {
      const cursor = url.searchParams.get('cursor');
      outputReads.push(cursor);
      init?.signal?.throwIfAborted();
      return (await intercept?.(cursor)) ?? json(200, server.read(cursor));
    }
    return realFetch(input, init);
  });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  queryClient.clear();
  queryClient.setDefaultOptions(queryDefaults);
  vi.restoreAllMocks();
  accountLifecycle.invalidate();
});

const render = (node: ReactNode) =>
  act(() =>
    root.render(
      <QueryClientProvider client={queryClient}>
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>{node}</I18nextProvider>
        </ChakraProvider>
      </QueryClientProvider>
    )
  );

const instance = { createdAt: '', id: 'server-output-1', typeId: 'server-output' };
const viewProps = {
  instance,
  manifest: serverOutputWidgetManifest,
  presentation: 'expanded',
  region: 'bottom',
  runtime: { workbench: { closeWidgetInstance } } as unknown as WidgetRuntimeApi,
} as WidgetViewProps;

/** The bottom panel's header slots and body, as the frame lays them out. */
const panel = (
  <Box display="flex" flexDirection="column" h="260px" w="400px">
    <HStack justify="space-between">
      <ServerOutputHeaderLabel instanceId={instance.id} region="bottom" />
      <ServerOutputHeaderActions {...viewProps} />
    </HStack>
    <ServerOutputWidgetView {...viewProps} />
  </Box>
);

const log = () => page.getByRole('log', { name: 'Server console output' });
const status = () => page.getByRole('status');
const scroller = () => log().element() as HTMLElement;
const rowTop = (id: number) =>
  scroller().querySelector(`[data-row="${id}"]`)!.getBoundingClientRect().top - scroller().getBoundingClientRect().top;
const isAtEnd = () => scroller().scrollHeight - scroller().scrollTop - scroller().clientHeight <= 4;
const wait = (ms: number) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
const lines = (count: number, from = 1) => Array.from({ length: count }, (_, index) => `line ${index + from}`);
const scrollRowToTop = (id: number) =>
  act(() => {
    scroller().scrollTop = rowTop(id) + scroller().scrollTop;
    scroller().dispatchEvent(new Event('scroll'));
  });

describe('Server Output discovery', () => {
  const DiscoveredBottomWidgets = () => (
    <ul>
      {getWidgetsForRegion('bottom', useWidgetCapabilities()).map((widget) => (
        <li key={widget.manifest.id}>{widget.manifest.id}</li>
      ))}
    </ul>
  );

  it('offers the widget only while the server grants it, loading the grant itself', async () => {
    config = { can_view_server_output: false, show_donation_link: false };
    await render(<DiscoveredBottomWidgets />);
    // Settled, not merely sent: an invalidation joins a first read still in flight.
    await vi.waitFor(() =>
      expect(queryClient.getQueryCache().findAll({ queryKey: ['frontend-config'] })[0]?.state.status).toBe('success')
    );
    await expect.element(page.getByText('server-status')).toBeVisible();
    expect(page.getByText('server-output').query()).toBeNull();

    config = { can_view_server_output: true, show_donation_link: false };
    await act(() => refetchFrontendConfig());
    await expect.element(page.getByText('server-output')).toBeVisible();

    // An older server omits the field: unavailable.
    config = { show_donation_link: false };
    await act(() => refetchFrontendConfig());
    await expect.poll(() => page.getByText('server-output').query()).toBeNull();
  });

  it('never offers it to an ordinary multi-user account, and never asks the server for one', async () => {
    setSession(multiuser(true));
    await render(<DiscoveredBottomWidgets />);
    await expect.element(page.getByText('server-output')).toBeVisible();

    // The session learns of a demotion: the session alone already denies.
    await act(() => setSession(multiuser(false)));
    expect(page.getByText('server-output').query()).toBeNull();
    expect(configReads).toBe(1);
  });
});

describe('Server Output panel', () => {
  it('refuses a restored widget the server does not grant, without reading, and offers a way out', async () => {
    config = { can_view_server_output: false, show_donation_link: false };
    await render(panel);

    await expect.element(page.getByText('This server is not capturing its console output.')).toBeVisible();
    await expect.element(status()).toHaveTextContent('Unavailable');
    // Nothing can be read, so there is nothing to copy.
    expect(page.getByRole('button', { name: 'Copy output' }).query()).toBeNull();
    await wait(150);
    expect(outputReads).toEqual([]);

    await page.getByRole('button', { name: 'Remove from layout' }).click();
    expect(closeWidgetInstance).toHaveBeenCalledWith(instance.id);
  });

  it('refuses an ordinary multi-user account before asking the server anything', async () => {
    setSession(multiuser(false));
    await render(panel);

    await expect.element(page.getByText("Only administrators can view the server's console output.")).toBeVisible();
    expect(page.getByRole('button', { name: 'Check again' }).query()).toBeNull();
    await wait(150);
    expect([configReads, outputReads.length]).toEqual([0, 0]);
  });

  it('recovers from a failed grant read when asked to check again', async () => {
    config = 'fail';
    server.append('alpha');
    await render(panel);
    await expect.element(page.getByRole('button', { name: 'Check again' })).toBeVisible();
    expect(outputReads).toEqual([]);

    config = { can_view_server_output: true, show_donation_link: false };
    await page.getByRole('button', { name: 'Check again' }).click();
    await expect.element(page.getByText('alpha')).toBeVisible();
  });

  it('shows an empty console, then output as it arrives, following the end, with one status word', async () => {
    await render(panel);
    await expect.element(page.getByText('No output yet')).toBeVisible();
    await expect.element(status()).toHaveTextContent('Live');
    await expect.element(page.getByRole('button', { name: 'Copy output' })).toBeDisabled();

    server.append(...lines(40));
    await expect.element(page.getByText('line 40', { exact: true })).toBeVisible();
    await vi.waitFor(() => expect(isAtEnd()).toBe(true));
    // The live region carries the state only, never output or counts.
    expect(status().element().textContent).toBe('Live');
    await expect.element(page.getByRole('button', { name: 'Copy output' })).toBeEnabled();

    // A progress row rewritten in place stays one row.
    server.append('Denoising 10%');
    await expect.element(page.getByText('Denoising 10%')).toBeVisible();
    server.rewrite(41, 'Denoising 90%');
    await expect.element(page.getByText('Denoising 90%')).toBeVisible();
    expect(page.getByText('Denoising 10%').query()).toBeNull();
    expect(isAtEnd()).toBe(true);
  });

  it('holds an older line in place while rows arrive and are evicted, until Jump to latest', async () => {
    server.append(...lines(200));
    await render(panel);
    await expect.element(page.getByText('line 200', { exact: true })).toBeVisible();
    await vi.waitFor(() => expect(isAtEnd()).toBe(true));

    await scrollRowToTop(100);
    const jump = page.getByRole('button', { name: 'Jump to latest' });
    await expect.element(jump).toBeVisible();
    const before = rowTop(100);

    server.append(...lines(60, 201));
    server.evictBelow(40);
    await expect.element(page.getByText('line 260', { exact: true })).toBeInTheDocument();
    await vi.waitFor(() => expect(scroller().querySelector('[data-row="39"]')).toBeNull());
    expect(Math.abs(rowTop(100) - before)).toBeLessThan(1);
    await expect.element(page.getByText('Earlier output is no longer available.')).toBeInTheDocument();

    await jump.click();
    await vi.waitFor(() => expect(isAtEnd()).toBe(true));
    await expect.element(log()).toHaveFocus();
    expect(jump.query()).toBeNull();
  });

  it('keeps the line being read in place when wrapping is turned off and on, and scrolls sideways unwrapped', async () => {
    server.append(...Array.from({ length: 150 }, (_, index) => `line ${index + 1} ${'long words '.repeat(12)}`));
    await render(panel);
    await expect.element(page.getByText(/^line 150 /)).toBeVisible();
    await vi.waitFor(() => expect(isAtEnd()).toBe(true));
    expect(scroller().scrollWidth).toBeLessThanOrEqual(scroller().clientWidth + 1);

    await scrollRowToTop(80);
    await act(() => setWrapLines(false));
    await vi.waitFor(() => expect(Math.abs(rowTop(80))).toBeLessThan(1));
    expect(scroller().scrollWidth).toBeGreaterThan(scroller().clientWidth);

    await act(() => setWrapLines(true));
    await vi.waitFor(() => expect(Math.abs(rowTop(80))).toBeLessThan(1));

    await page.getByRole('button', { name: 'Jump to latest' }).click();
    await vi.waitFor(() => expect(isAtEnd()).toBe(true));
    await act(() => setWrapLines(false));
    await vi.waitFor(() => expect(isAtEnd()).toBe(true));
  });

  it('keeps its sideways extent while long rows scroll out of view, wrapping off', async () => {
    setWrapLines(false);
    server.append(
      ...Array.from({ length: 64 }, (_, index) => `traceback ${index + 1} ${'x'.repeat(600)}`),
      ...lines(400, 65)
    );
    await render(panel);
    await expect.element(page.getByText('line 464', { exact: true })).toBeVisible();
    await vi.waitFor(() => expect(isAtEnd()).toBe(true));

    // Only short rows are on screen; the long ones still set how far the log scrolls sideways.
    expect(scroller().scrollWidth).toBeGreaterThan(scroller().clientWidth * 3);

    await act(() => {
      scroller().scrollTop = 0;
      scroller().scrollLeft = scroller().scrollWidth;
    });
    const right = scroller().scrollLeft;
    expect(right).toBeGreaterThan(scroller().clientWidth * 2);
    await act(() => {
      scroller().scrollTop = scroller().scrollHeight;
    });
    await wait(50);
    expect(scroller().scrollLeft).toBe(right);
  });

  it('selects and copies the output from the keyboard', async () => {
    server.append('alpha', 'beta', 'gamma');
    await render(panel);
    await expect.element(page.getByText('gamma')).toBeVisible();
    const copied: string[] = [];
    const onCopy = (event: ClipboardEvent) => copied.push(event.clipboardData?.getData('text/plain') ?? '');
    document.addEventListener('copy', onCopy);

    try {
      await userEvent.click(log());
      // Nothing selected: the whole transcript.
      await userEvent.keyboard('{Control>}c{/Control}');
      // Select-all stays inside the log.
      await userEvent.keyboard('{Control>}a{/Control}');
      expect(document.getSelection()?.toString()).not.toContain('Live');
      await userEvent.keyboard('{Control>}c{/Control}');

      expect(copied).toEqual(['alpha\nbeta\ngamma', 'alpha\nbeta\ngamma']);
    } finally {
      document.removeEventListener('copy', onCopy);
    }
  });

  it('copies a partial selection across styled spans and rows as the transcript holds it', async () => {
    server.append(
      { styles: [{ fg: '6', length: 4, offset: 7 }], text: '[time] INFO --> first' },
      { styles: [{ bold: true, length: 6, offset: 0 }], text: 'second row' },
      { styles: [{ fg: '1', length: 5, offset: 6 }], text: 'third ERROR tail' }
    );
    await render(panel);
    await expect.element(page.getByText('tail', { exact: false })).toBeVisible();
    const copied: string[] = [];
    const onCopy = (event: ClipboardEvent) => copied.push(event.clipboardData?.getData('text/plain') ?? '');
    document.addEventListener('copy', onCopy);

    try {
      const content = scroller().querySelector('[data-row="1"]')!.parentElement!.parentElement!;
      const info = [...content.querySelectorAll('[data-row="1"] span')][0]!.firstChild!; // "INFO"
      const error = [...content.querySelectorAll('[data-row="3"] span')][0]!.firstChild!; // "ERROR"
      const range = document.createRange();
      range.setStart(info, 2);
      range.setEnd(error, 3);
      document.getSelection()!.removeAllRanges();
      document.getSelection()!.addRange(range);
      log().element().focus();
      await userEvent.keyboard('{Control>}c{/Control}');

      expect(copied).toEqual(['FO --> first\nsecond row\nthird ERR']);
    } finally {
      document.removeEventListener('copy', onCopy);
    }
  });

  it('stops reading once closed', async () => {
    server.append('alpha');
    await render(panel);
    await expect.element(page.getByText('alpha')).toBeVisible();
    await vi.waitFor(() => expect(outputReads.length).toBeGreaterThan(1));

    await act(() => root.render(null));
    const count = outputReads.length;
    await wait(200);
    expect(outputReads).toHaveLength(count);
  });

  it('clears on an account change and ignores the old account’s late response', async () => {
    server.append('alpha');
    await render(panel);
    await expect.element(page.getByText('alpha')).toBeVisible();

    let release: (() => void) | null = null;
    intercept = () =>
      new Promise<Response>((resolve) => {
        server.append('from the old account');
        release = () => resolve(json(200, server.read(outputReads.at(-2) ?? null)));
      });
    await vi.waitFor(() => expect(release).not.toBeNull());

    await act(() => accountLifecycle.activate('another-account'));
    expect(page.getByText('alpha').query()).toBeNull();

    await act(() => release!());
    await wait(100);
    expect(page.getByText('from the old account').query()).toBeNull();
  });

  it('says when the server restarted or history expired, until dismissed', async () => {
    server.append('before restart');
    await render(panel);
    await expect.element(page.getByText('before restart')).toBeVisible();

    server.restart();
    server.append('after restart');
    await expect.element(page.getByText('after restart')).toBeVisible();
    const restarted = page.getByText('The server restarted. Output from before the restart is no longer available.');
    await expect.element(restarted).toBeVisible();
    expect(page.getByText('before restart').query()).toBeNull();

    await page.getByRole('button', { name: 'Dismiss' }).click();
    expect(restarted.query()).toBeNull();
    await expect.element(log()).toHaveFocus();

    // A later replacement is a new notice.
    server.expire();
    server.append('latest');
    await expect
      .element(page.getByText('Some output was missed while this view fell behind. Showing the latest output.'))
      .toBeVisible();
  });

  it('waits for the server rather than claiming an empty console when the first read fails', async () => {
    intercept = () => Promise.resolve(json(503, { detail: 'Unavailable' }));
    await render(panel);

    await expect.element(page.getByText("Can't reach the server yet. Retrying…")).toBeVisible();
    await expect.element(status()).toHaveTextContent('Reconnecting');
    expect(page.getByText('No output yet').query()).toBeNull();

    intercept = null;
    server.append('alpha');
    await expect.element(page.getByText('alpha')).toBeVisible();
    await expect.element(status()).toHaveTextContent('Live');
  });

  it('reconnects after a transient failure and resumes from its cursor', async () => {
    server.append('alpha');
    await render(panel);
    await expect.element(page.getByText('alpha')).toBeVisible();

    intercept = () => Promise.resolve(json(503, { detail: 'Unavailable' }));
    await expect.element(status()).toHaveTextContent('Reconnecting');
    await expect.element(page.getByText('alpha')).toBeVisible();

    const failedAt = outputReads.length;
    intercept = null;
    server.append('beta');
    await expect.element(page.getByText('beta')).toBeVisible();
    expect(outputReads[failedAt]).not.toBeNull();
    await expect.element(status()).toHaveTextContent('Live');
  });

  it('says capture is off when a single-user server stops capturing, clearing the output', async () => {
    server.append('secret');
    await render(panel);
    await expect.element(page.getByText('secret')).toBeVisible();
    const grantReads = configReads;

    intercept = () => Promise.resolve(json(404, { detail: 'Server output is not available' }));
    config = { can_view_server_output: false, show_donation_link: false };
    await expect.element(page.getByText('This server is not capturing its console output.')).toBeVisible();
    expect(page.getByText('secret').query()).toBeNull();
    await vi.waitFor(() => expect(configReads).toBeGreaterThan(grantReads));
    const count = outputReads.length;
    await wait(150);
    expect(outputReads).toHaveLength(count);
  });

  it('keeps one refusal for a demoted admin, then reads afresh once the grant is back', async () => {
    setSession(multiuser(true));
    server.append('secret');
    await render(panel);
    await expect.element(page.getByText('secret')).toBeVisible();
    const grantReads = configReads;

    // Another admin demotes this one: the session still says admin; the server refuses and its config denies.
    intercept = () => Promise.resolve(json(403, { detail: 'Admin privileges required' }));
    config = { can_view_server_output: false, show_donation_link: false };
    const forbidden = page.getByText("Only administrators can view the server's console output.");
    await expect.element(forbidden).toBeVisible();
    expect(page.getByText('secret').query()).toBeNull();
    await expect.element(status()).toHaveTextContent('Unavailable');
    expect(page.getByRole('button', { name: 'Copy output' }).query()).toBeNull();

    // The config answers after the refusal; the message does not change under the reader.
    await vi.waitFor(() => expect(configReads).toBeGreaterThan(grantReads));
    await wait(100);
    await expect.element(forbidden).toBeVisible();
    const count = outputReads.length;
    await wait(150);
    expect(outputReads).toHaveLength(count);

    // Promoted again: checking again mounts a fresh reader.
    intercept = null;
    config = { can_view_server_output: true, show_donation_link: false };
    server.append('welcome back');
    await page.getByRole('button', { name: 'Check again' }).click();
    await expect.element(page.getByText('welcome back')).toBeVisible();
    expect(outputReads.at(-1) === null || outputReads.includes(null)).toBe(true);
  });
});
