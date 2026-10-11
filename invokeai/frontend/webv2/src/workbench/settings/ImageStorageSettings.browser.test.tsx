import type * as GalleryQueryCacheModule from '@features/gallery/data/queryCache';
import type * as HttpModule from '@platform/transport/http';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ImageStorageSettings } from './ImageStorageSettings';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn<(path: string, init?: RequestInit) => Promise<unknown>>(),
  canManageAppConfig: true,
  invalidateGallery: vi.fn((..._args: unknown[]) => Promise.resolve()),
}));

vi.mock('@features/identity', () => ({ useCapabilities: () => ({ canManageAppConfig: mocks.canManageAppConfig }) }));
vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<typeof HttpModule>()),
  apiFetchJson: (path: string, init?: RequestInit) => mocks.apiFetchJson(path, init),
}));
vi.mock('@features/gallery/data/queryCache', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryQueryCacheModule>()),
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...args),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => [key, ...Object.values(values ?? {})].join(' '),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const STATUS_ROUTE = '/api/v1/image_moves/status';
const CONFIG_ROUTE = '/api/v1/app/runtime_config';

const idleStatus = (needsMoveCount: number) => ({
  active_job_id: null,
  is_running: false,
  last_error: null,
  latest_job: needsMoveCount === 0 ? { error_message: null, id: 2, state: 'committed' } : null,
  needs_move_count: needsMoveCount,
  operation: null,
});
const runningStatus = (operation: 'move_all' | 'recovery') => ({
  active_job_id: 3,
  is_running: true,
  last_error: null,
  latest_job: { error_message: null, id: 3, state: 'moving' },
  needs_move_count: 12,
  operation,
});
const interruptedStatus = {
  active_job_id: 3,
  is_running: false,
  last_error: 'Server stopped during an image move',
  latest_job: { error_message: null, id: 3, state: 'moving' },
  needs_move_count: 12,
  operation: null,
};

/** Routes answer from a mutable table so a test can change what the next poll sees. */
const routes = new Map<string, () => unknown>();
const installRoutes = (table: Record<string, () => unknown>) => {
  routes.clear();
  for (const [key, answer] of Object.entries(table)) {
    routes.set(key, answer);
  }
  mocks.apiFetchJson.mockImplementation((path, init) => {
    const answer = routes.get(`${init?.method ?? 'GET'} ${path}`);

    if (!answer) {
      return Promise.reject(new Error(`No route for ${init?.method ?? 'GET'} ${path}`));
    }
    const value = answer();

    return value instanceof Error ? Promise.reject(value) : Promise.resolve(value);
  });
};
const calls = (method: string, path: string) =>
  mocks.apiFetchJson.mock.calls.filter(
    ([calledPath, init]) => calledPath === path && (init?.method ?? 'GET') === method
  );

let host: HTMLDivElement;
let root: Root;
let queryClient: QueryClient;

const render = async () => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <ImageStorageSettings />
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
};
const text = () => host.textContent ?? '';
const button = (label: string) =>
  [...host.querySelectorAll<HTMLButtonElement>('button')].find((candidate) => candidate.textContent?.trim() === label);
const dialog = () => document.querySelector<HTMLElement>('[role="alertdialog"]');
const confirmIn = (label: string) =>
  [...(dialog()?.querySelectorAll<HTMLButtonElement>('button') ?? [])].find((candidate) =>
    candidate.textContent?.includes(label)
  );
const trigger = () => host.querySelector<HTMLButtonElement>('[data-scope="select"][data-part="trigger"]');
const refetchStatus = () => act(() => queryClient.refetchQueries({ queryKey: ['gallery', 'image-storage'] }));

beforeEach(() => {
  vi.clearAllMocks();
  mocks.canManageAppConfig = true;
  accountLifecycle.activate('admin');
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  queryClient.clear();
});

describe('image storage settings', () => {
  it('shows a read-only notice to non-admins without reading admin-only routes', async () => {
    mocks.canManageAppConfig = false;
    await render();

    expect(text()).toContain('settings.imageStorage.adminOnly');
    expect(mocks.apiFetchJson).not.toHaveBeenCalled();
  });

  it('reads the layout and move status, and patches only the layout when it changes', async () => {
    let layout = 'flat';
    installRoutes({
      [`GET ${CONFIG_ROUTE}`]: () => ({ config: { image_subfolder_strategy: layout }, set_fields: [] }),
      [`PATCH ${CONFIG_ROUTE}`]: () => ({ config: { image_subfolder_strategy: layout }, set_fields: [] }),
      [`GET ${STATUS_ROUTE}`]: () => idleStatus(layout === 'flat' ? 0 : 12),
    });
    await render();

    await expect.poll(() => trigger()?.textContent).toContain('settings.imageStorage.layouts.flat');
    await expect.poll(() => text()).toContain('settings.imageStorage.statusIdle 0');
    expect(text()).toContain('settings.imageStorage.latestJob 2 settings.imageStorage.jobStates.committed');

    await act(() => userEvent.click(trigger()!));
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBe(4);
    const dateOption = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((option) =>
      option.textContent?.includes('settings.imageStorage.layouts.date')
    );
    layout = 'date';
    await act(() => userEvent.click(dateOption!));

    await expect.poll(() => calls('PATCH', CONFIG_ROUTE).length).toBe(1);
    expect(calls('PATCH', CONFIG_ROUTE)[0]?.[1]?.body).toBe(JSON.stringify({ image_subfolder_strategy: 'date' }));
    await expect.poll(() => trigger()?.textContent).toContain('settings.imageStorage.layouts.date');
    // The count of images outside the layout is re-read after the change.
    await expect.poll(() => text()).toContain('settings.imageStorage.statusIdle 12');
  });

  it('keeps the server layout and shows the refusal when the layout change is rejected', async () => {
    installRoutes({
      [`GET ${CONFIG_ROUTE}`]: () => ({ config: { image_subfolder_strategy: 'flat' } }),
      [`PATCH ${CONFIG_ROUTE}`]: () => new ApiError(JSON.stringify({ detail: 'Admin privileges required' }), 403),
      [`GET ${STATUS_ROUTE}`]: () => idleStatus(0),
    });
    await render();
    await expect.poll(() => trigger()?.textContent).toContain('settings.imageStorage.layouts.flat');

    await act(() => userEvent.click(trigger()!));
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBe(4);
    const hashOption = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((option) =>
      option.textContent?.includes('settings.imageStorage.layouts.hash')
    );
    await act(() => userEvent.click(hashOption!));

    await expect.poll(() => host.querySelector('[role="alert"]')?.textContent).toBe('Admin privileges required');
    expect(trigger()?.textContent).toContain('settings.imageStorage.layouts.flat');
    expect(trigger()?.disabled).toBe(false);
    expect(calls('GET', STATUS_ROUTE)).toHaveLength(1);
  });

  it('shows a layout this build does not know as itself instead of reading as flat', async () => {
    installRoutes({
      [`GET ${CONFIG_ROUTE}`]: () => ({ config: { image_subfolder_strategy: 'by_user' } }),
      [`GET ${STATUS_ROUTE}`]: () => idleStatus(0),
    });
    await render();

    await expect.poll(() => trigger()?.textContent).toContain('settings.imageStorage.layouts.unknown by_user');
    await act(() => userEvent.click(trigger()!));
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBe(5);
    expect(calls('PATCH', CONFIG_ROUTE)).toHaveLength(0);
  });

  it('starts a move after confirmation, polls it, and refreshes the gallery once it stops', async () => {
    let status: unknown = idleStatus(12);
    installRoutes({
      [`GET ${CONFIG_ROUTE}`]: () => ({ config: { image_subfolder_strategy: 'date' } }),
      [`GET ${STATUS_ROUTE}`]: () => status,
      [`POST /api/v1/image_moves/start`]: () => {
        status = runningStatus('move_all');
        return status;
      },
    });
    await render();
    await expect.poll(() => button('settings.imageStorage.startMove')?.disabled).toBe(false);

    await act(() => userEvent.click(button('settings.imageStorage.startMove')!));
    await expect.poll(() => dialog()).not.toBeNull();
    expect(dialog()?.textContent).toContain('settings.imageStorage.startConfirmTitle');
    expect(calls('POST', '/api/v1/image_moves/start')).toHaveLength(0);

    await act(() => userEvent.click(confirmIn('settings.imageStorage.startMove')!));
    await expect.poll(() => calls('POST', '/api/v1/image_moves/start').length).toBe(1);
    await expect.poll(() => text()).toContain('settings.imageStorage.statusMoving');
    expect(button('settings.imageStorage.startMove')?.disabled).toBe(true);
    expect(button('settings.imageStorage.recover')?.disabled).toBe(true);
    expect(trigger()?.disabled).toBe(true);
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();

    status = idleStatus(0);
    await refetchStatus();

    await expect.poll(() => text()).toContain('settings.imageStorage.statusIdle 0');
    expect(mocks.invalidateGallery).toHaveBeenCalledTimes(1);
    expect(button('settings.imageStorage.startMove')?.disabled).toBe(false);
    expect(trigger()?.disabled).toBe(false);
  });

  it('offers recovery for an interrupted job and reports a refused start', async () => {
    let status: unknown = interruptedStatus;
    installRoutes({
      [`GET ${CONFIG_ROUTE}`]: () => ({ config: { image_subfolder_strategy: 'date' } }),
      [`GET ${STATUS_ROUTE}`]: () => status,
      [`POST /api/v1/image_moves/recover`]: () => {
        status = runningStatus('recovery');
        return status;
      },
    });
    await render();

    await expect.poll(() => text()).toContain('settings.imageStorage.statusNeedsRecovery 3');
    expect(text()).toContain('Server stopped during an image move');
    expect(button('settings.imageStorage.startMove')?.disabled).toBe(true);
    expect(button('settings.imageStorage.recover')?.disabled).toBe(false);

    await act(() => userEvent.click(button('settings.imageStorage.recover')!));
    await expect.poll(() => dialog()).not.toBeNull();
    await act(() => userEvent.click(confirmIn('settings.imageStorage.recover')!));

    await expect.poll(() => calls('POST', '/api/v1/image_moves/recover').length).toBe(1);
    await expect.poll(() => text()).toContain('settings.imageStorage.statusRecovering');
  });

  it('shows the server refusal when a move cannot start, and leaves the controls usable', async () => {
    installRoutes({
      [`GET ${CONFIG_ROUTE}`]: () => ({ config: { image_subfolder_strategy: 'date' } }),
      [`GET ${STATUS_ROUTE}`]: () => idleStatus(12),
      [`POST /api/v1/image_moves/start`]: () => new ApiError(JSON.stringify({ detail: 'Queue work is active' }), 409),
    });
    await render();
    await expect.poll(() => button('settings.imageStorage.startMove')?.disabled).toBe(false);

    await act(() => userEvent.click(button('settings.imageStorage.startMove')!));
    await expect.poll(() => dialog()).not.toBeNull();
    await act(() => userEvent.click(confirmIn('settings.imageStorage.startMove')!));

    await expect.poll(() => host.querySelector('[role="alert"]')?.textContent).toBe('Queue work is active');
    expect(button('settings.imageStorage.startMove')?.disabled).toBe(false);
    expect(button('settings.imageStorage.startMove')?.getAttribute('aria-describedby')).toBe(
      host.querySelector('[role="alert"]')?.id
    );
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
  });

  it('shows a load failure for the status with a retry', async () => {
    let fail = true;
    installRoutes({
      [`GET ${CONFIG_ROUTE}`]: () => ({ config: { image_subfolder_strategy: 'flat' } }),
      [`GET ${STATUS_ROUTE}`]: () =>
        fail ? new ApiError('{"detail":"Image move service unavailable"}', 503) : idleStatus(0),
    });
    await render();

    await expect.poll(() => host.querySelector('[role="alert"]')?.textContent).toBe('Image move service unavailable');
    expect(button('settings.imageStorage.startMove')?.disabled).toBe(true);

    fail = false;
    await act(() => userEvent.click(button('common.retry')!));

    await expect.poll(() => text()).toContain('settings.imageStorage.statusIdle 0');
    expect(button('settings.imageStorage.startMove')?.disabled).toBe(false);
  });
});
