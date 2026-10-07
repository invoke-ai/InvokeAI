import type * as GalleryQueriesModule from '@features/gallery/queries';
import type * as IdentityModule from '@features/identity';
import type * as HttpModule from '@platform/transport/http';
import type * as ImageMapStoreModule from '@workbench/image-map/imageMapStore';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type * as SettingsStoreModule from './store';

import { WorkspaceSettings } from './CustomSettingsEditors';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
  canManageAppConfig: true,
  invalidateGallery: vi.fn(),
  refreshImageMapPoints: vi.fn(),
}));

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<typeof IdentityModule>()),
  useCapabilities: () => ({ canManageAppConfig: mocks.canManageAppConfig }),
}));
vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<typeof HttpModule>()),
  apiFetchJson: (...args: unknown[]) => mocks.apiFetchJson(...args),
}));
vi.mock('@features/gallery/queries', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryQueriesModule>()),
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...args),
}));
vi.mock('@workbench/image-map/imageMapStore', async (importOriginal) => ({
  ...(await importOriginal<typeof ImageMapStoreModule>()),
  refreshImageMapPoints: (...args: unknown[]) => mocks.refreshImageMapPoints(...args),
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => null,
  useActiveProjectSelector: (selector: (project: { queue: { items: never[] } }) => unknown) =>
    selector({ queue: { items: [] } }),
  useOptionalWorkbenchCommands: () => null,
  useOptionalWorkbenchPersistenceService: () => null,
}));
vi.mock('@tanstack/react-router', () => ({ useNavigate: () => vi.fn() }));
vi.mock('@workbench/palette/PaletteButton', () => ({ PaletteButton: () => null }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('@workbench/shell/topbar/useTopbarShortcut', () => ({ useTopbarShortcut: () => null }));
vi.mock('./store', async (importOriginal) => ({
  ...(await importOriginal<typeof SettingsStoreModule>()),
  useWorkbenchSettingsSelector: (selector: (snapshot: never) => unknown) => selector({ scope: 'user' } as never),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => [key, ...Object.values(values ?? {})].join(' '),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const operations = [
  {
    operation: 'remove_missing',
    button: 'settings.galleryMaintenance.removeMissing',
    description: 'settings.galleryMaintenance.removeMissingDescription',
    title: 'settings.galleryMaintenance.removeMissingTitle',
    confirm: 'settings.galleryMaintenance.removeMissingConfirm',
    route: '/api/v1/app/gallery/maintenance/remove-missing',
    fingerprint: '1'.repeat(64),
  },
  {
    operation: 'archive_untracked',
    button: 'settings.galleryMaintenance.archiveUntracked',
    description: 'settings.galleryMaintenance.archiveUntrackedDescription',
    title: 'settings.galleryMaintenance.archiveUntrackedTitle',
    confirm: 'settings.galleryMaintenance.archiveUntrackedConfirm',
    route: '/api/v1/app/gallery/maintenance/archive-untracked',
    fingerprint: '2'.repeat(64),
  },
  {
    operation: 'regenerate_thumbnails',
    button: 'settings.galleryMaintenance.regenerateThumbnails',
    description: 'settings.galleryMaintenance.regenerateThumbnailsDescription',
    title: 'settings.galleryMaintenance.regenerateThumbnailsTitle',
    confirm: 'settings.galleryMaintenance.regenerateThumbnailsConfirm',
    route: '/api/v1/app/gallery/maintenance/regenerate-thumbnails',
    fingerprint: '3'.repeat(64),
  },
] as const;

const preview = (operation: string, fingerprint: string, affectedCount = 2) => ({
  operation,
  fingerprint,
  examined_count: 12,
  affected_count: affectedCount,
  skipped_count: 1,
  error_count: 0,
  errors: [],
  archive_path: '/tmp/invokeai/gallery-archive/run-1',
});

const result = (
  operation: string,
  overrides: Partial<{
    status: 'completed' | 'partial' | 'no_op' | 'failed';
    records_removed: number;
    images_archived: number;
    thumbnails_archived: number;
    thumbnails_regenerated: number;
    failed_count: number;
    archive_path: string | null;
    backup_path: string | null;
  }> = {}
) => ({
  operation,
  status: 'completed' as const,
  examined_count: 12,
  skipped_count: 1,
  failed_count: 0,
  records_removed: 0,
  images_archived: 0,
  thumbnails_archived: 0,
  thumbnails_regenerated: 0,
  archive_path: null,
  backup_path: '/tmp/invokeai/gallery-backups/run-1.sqlite',
  errors: [],
  ...overrides,
});

describe('gallery maintenance in Data & workspace settings', () => {
  let host: HTMLDivElement;
  let root: Root;
  let queryClient: QueryClient;

  const renderSettings = async () => {
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <WorkspaceSettings onReveal={() => {}} />
          </QueryClientProvider>
        </ChakraProvider>
      )
    );
  };

  const getButton = (key: string) =>
    [...document.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent?.includes(key));

  const getConfirmButton = (action: (typeof operations)[number]) =>
    [...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')].find((button) =>
      button.textContent?.includes(action.confirm)
    );

  const openAction = async (action: (typeof operations)[number]) => {
    await expect.poll(() => getButton(action.button)).toBeDefined();
    await act(() => userEvent.click(getButton(action.button)!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).not.toBeNull();
    await expect.poll(() => getConfirmButton(action)?.disabled).toBe(false);
    const background = getComputedStyle(getConfirmButton(action)!).backgroundColor;
    const [red = 0, green = 0, blue = 0] = background.match(/\d+/g)?.map(Number) ?? [];
    expect(red).toBeGreaterThan(green);
    expect(red).toBeGreaterThan(blue);
  };

  beforeEach(() => {
    mocks.canManageAppConfig = true;
    mocks.apiFetchJson.mockReset();
    mocks.invalidateGallery.mockReset().mockResolvedValue(undefined);
    mocks.refreshImageMapPoints.mockReset().mockResolvedValue(undefined);
    accountLifecycle.activate('gallery-maintenance-settings-test', ':user:gallery-maintenance-settings-test');
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    queryClient.clear();
    accountLifecycle.invalidate();
  });

  it('shows three described administrator actions and hides them from non-admins', async () => {
    await renderSettings();

    for (const action of operations) {
      await expect.poll(() => getButton(action.button)).toBeDefined();
      expect(getButton(action.button)).toBeDefined();
      expect(host.textContent).toContain(action.description);
    }
    expect(host.textContent).toContain('settings.databaseMaintenance.compactDatabase');
    expect(host.textContent).toContain('settings.catalog.manageIntermediates');
    expect(host.textContent).toContain('Clear saved data');
    expect(mocks.apiFetchJson).not.toHaveBeenCalled();

    mocks.canManageAppConfig = false;
    await renderSettings();
    for (const action of operations) {
      expect(getButton(action.button)).toBeUndefined();
    }
    expect(host.textContent).toContain('Clear saved data');
  });

  it.each(operations)('$operation previews before confirmation and cancel sends no execute request', async (action) => {
    mocks.apiFetchJson.mockResolvedValueOnce(preview(action.operation, action.fingerprint));
    await renderSettings();
    await openAction(action);

    expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain(action.title);
    expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('2');
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/app/gallery/maintenance/preview', {
      method: 'POST',
      body: JSON.stringify({ operation: action.operation }),
      signal: expect.any(AbortSignal),
    });

    await act(() => userEvent.keyboard('{Escape}'));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('keeps confirmation disabled during preview, shows preview errors, and permits a read-only retry', async () => {
    let resolvePreview!: (value: ReturnType<typeof preview>) => void;
    mocks.apiFetchJson.mockRejectedValueOnce(new Error('Scan could not read the image directory')).mockReturnValueOnce(
      new Promise((resolve) => {
        resolvePreview = resolve;
      })
    );
    await renderSettings();
    await act(() => userEvent.click(getButton(operations[0].button)!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).not.toBeNull();
    await expect
      .poll(() => document.querySelector('[role="alert"]')?.textContent)
      .toContain('Scan could not read the image directory');
    expect(getConfirmButton(operations[0])?.disabled).toBe(true);
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

    await act(() => userEvent.click(getButton('settings.galleryMaintenance.retryPreview')!));
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
    expect(getConfirmButton(operations[0])?.disabled).toBe(true);
    await act(() => resolvePreview(preview(operations[0].operation, operations[0].fingerprint, 0)));
    await expect.poll(() => getConfirmButton(operations[0])?.disabled).toBe(false);
    expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('0');
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
  });

  it('executes once, retains partial results, and refreshes gallery and map after record removal', async () => {
    const owner = accountLifecycle.capture();
    mocks.apiFetchJson
      .mockResolvedValueOnce(preview('remove_missing', operations[0].fingerprint))
      .mockResolvedValueOnce(result('remove_missing', { status: 'partial', records_removed: 3, failed_count: 1 }));
    await renderSettings();
    await openAction(operations[0]);
    await act(() => userEvent.click(getConfirmButton(operations[0])!));

    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
    expect(mocks.apiFetchJson).toHaveBeenLastCalledWith(operations[0].route, {
      method: 'POST',
      body: JSON.stringify({ fingerprint: operations[0].fingerprint }),
      signal: expect.any(AbortSignal),
    });
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(host.textContent).toContain('settings.galleryMaintenance.partial');
    expect(host.textContent).toContain('3');
    expect(host.textContent).toContain('/tmp/invokeai/gallery-backups/run-1.sqlite');
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    expect(mocks.invalidateGallery).toHaveBeenCalledWith(queryClient, owner);
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledOnce();
  });

  it('refreshes the semantic map even when gallery invalidation fails after committed removals', async () => {
    mocks.apiFetchJson
      .mockResolvedValueOnce(preview('remove_missing', operations[0].fingerprint))
      .mockResolvedValueOnce(result('remove_missing', { status: 'partial', records_removed: 2, failed_count: 1 }));
    mocks.invalidateGallery.mockRejectedValueOnce(new Error('Gallery refresh failed'));
    await renderSettings();
    await openAction(operations[0]);
    await act(() => userEvent.click(getConfirmButton(operations[0])!));

    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledOnce();
    expect(host.textContent).toContain('settings.galleryMaintenance.partial');
    expect(host.textContent).toContain('Gallery refresh failed');
  });

  it('keeps execution open and blocks dismissal and duplicate actions while the server is working', async () => {
    let resolveExecution!: (value: ReturnType<typeof result>) => void;
    mocks.apiFetchJson
      .mockResolvedValueOnce(preview('archive_untracked', operations[1].fingerprint))
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveExecution = resolve;
        })
      );
    await renderSettings();
    await openAction(operations[1]);
    const confirmButton = getConfirmButton(operations[1])!;
    await act(() => userEvent.click(confirmButton));

    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
    expect(confirmButton.disabled).toBe(true);
    const cancelButton = [...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')].find(
      (button) => button.textContent?.includes('Cancel')
    )!;
    expect(cancelButton.disabled).toBe(true);
    expect(getButton(operations[0].button)?.disabled).toBe(true);
    await act(() => userEvent.keyboard('{Escape}'));
    expect(document.querySelector('[role="alertdialog"]')).not.toBeNull();

    await act(() => resolveExecution(result('archive_untracked', { images_archived: 4 })));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
  });

  it('fences a late preview when the account changes and requires a fresh scan', async () => {
    let resolvePreview!: (value: ReturnType<typeof preview>) => void;
    mocks.apiFetchJson
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolvePreview = resolve;
        })
      )
      .mockResolvedValueOnce(preview('remove_missing', '5'.repeat(64)));
    await renderSettings();
    await expect.poll(() => getButton(operations[0].button)).toBeDefined();
    await act(() => userEvent.click(getButton(operations[0].button)!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).not.toBeNull();

    await act(() =>
      accountLifecycle.activate('gallery-maintenance-next-account', ':user:gallery-maintenance-next-account')
    );
    await renderSettings();
    await act(async () => {
      resolvePreview(preview('remove_missing', operations[0].fingerprint));
      await Promise.resolve();
    });

    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(host.textContent).not.toContain('settings.galleryMaintenance.accountChanged');
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
    await act(() => userEvent.click(getButton(operations[0].button)!));
    await expect.poll(() => getConfirmButton(operations[0])?.disabled).toBe(false);
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
  });

  it('hides an execution result from the previous account and stays busy until that request settles', async () => {
    let resolveExecution!: (value: ReturnType<typeof result>) => void;
    mocks.apiFetchJson.mockResolvedValueOnce(preview('remove_missing', operations[0].fingerprint)).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveExecution = resolve;
      })
    );
    await renderSettings();
    await openAction(operations[0]);
    await act(() => userEvent.click(getConfirmButton(operations[0])!));

    await act(() =>
      accountLifecycle.activate('gallery-maintenance-third-account', ':user:gallery-maintenance-third-account')
    );
    expect(getButton(operations[0].button)?.disabled).toBe(true);
    await act(async () => {
      resolveExecution(result('remove_missing', { records_removed: 2 }));
      await Promise.resolve();
    });
    await expect.poll(() => getButton(operations[0].button)?.disabled).toBe(false);

    expect(host.querySelector('[data-testid="gallery-maintenance-result"]')).toBeNull();
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();
  });

  it('turns a changed-preview conflict into a new scan instead of repeating the mutation', async () => {
    mocks.apiFetchJson
      .mockResolvedValueOnce(preview('remove_missing', operations[0].fingerprint))
      .mockRejectedValueOnce(new ApiError('{"detail":"Preview has changed"}', 409))
      .mockResolvedValueOnce(preview('remove_missing', '4'.repeat(64), 1));
    await renderSettings();
    await openAction(operations[0]);
    await act(() => userEvent.click(getConfirmButton(operations[0])!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();

    expect(host.textContent).toContain('settings.galleryMaintenance.conflict');
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
    await act(() => userEvent.click(getButton('settings.galleryMaintenance.reviewCurrentItems')!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).not.toBeNull();
    await expect.poll(() => getConfirmButton(operations[0])?.disabled).toBe(false);
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(3);
  });

  it.each([operations[1], operations[2]])('$operation does not refresh embedding read models', async (action) => {
    mocks.apiFetchJson.mockResolvedValueOnce(preview(action.operation, action.fingerprint)).mockResolvedValueOnce(
      result(action.operation, {
        status: action.operation === 'regenerate_thumbnails' ? 'no_op' : 'completed',
        records_removed: 4,
        images_archived: action.operation === 'archive_untracked' ? 4 : 0,
        thumbnails_regenerated: action.operation === 'regenerate_thumbnails' ? 5 : 0,
      })
    );
    await renderSettings();
    await openAction(action);
    await act(() => userEvent.click(getConfirmButton(action)!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();

    expect(host.textContent).toContain(
      action.operation === 'regenerate_thumbnails'
        ? 'settings.galleryMaintenance.no_op'
        : 'settings.galleryMaintenance.completed'
    );
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();
  });
});
