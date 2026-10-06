import type * as IdentityModule from '@features/identity';
import type * as ConnectionStoreModule from '@platform/transport/connectionStore';
import type * as HttpModule from '@platform/transport/http';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type * as SettingsStoreModule from './store';

const mocks = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  canManageAppConfig: true,
  notifyError: vi.fn(),
  notifySuccess: vi.fn(),
}));

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<typeof IdentityModule>()),
  AccountMenu: () => null,
  AccountMenuSection: () => null,
  useCapabilities: () => ({
    canManageAppConfig: mocks.canManageAppConfig,
    canManageModels: false,
    canManageNodes: false,
  }),
  useHasAccountSection: () => false,
}));
vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<typeof HttpModule>()),
  apiFetch: (...args: unknown[]) => mocks.apiFetch(...args),
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => null,
  useActiveProjectSelector: (selector: (project: { queue: { items: never[] } }) => unknown) =>
    selector({ queue: { items: [] } }),
  useOptionalWorkbenchCommands: () => null,
  useOptionalWorkbenchPersistenceService: () => null,
}));
vi.mock('@platform/transport/connectionStore', async (importOriginal) => ({
  ...(await importOriginal<typeof ConnectionStoreModule>()),
  useConnectionStatusSelector: (selector: (snapshot: never) => unknown) => selector({ status: 'connected' } as never),
}));
vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => vi.fn(),
}));
vi.mock('@workbench/palette/PaletteButton', () => ({ PaletteButton: () => null }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('@workbench/shell/topbar/useTopbarShortcut', () => ({ useTopbarShortcut: () => null }));
vi.mock('./store', async (importOriginal) => ({
  ...(await importOriginal<typeof SettingsStoreModule>()),
  useWorkbenchSettingsSelector: (selector: (snapshot: never) => unknown) => selector({ scope: 'user' } as never),
}));
vi.mock('@workbench/useNotify', () => ({
  useNotify: () => ({ error: mocks.notifyError, success: mocks.notifySuccess }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { LaunchpadTopBar } from '@workbench/launchpad/LaunchpadTopBar';
import { AppMenu } from '@workbench/shell/topbar/AppMenu';

import { WorkspaceSettings } from './CustomSettingsEditors';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('database maintenance in Data & workspace settings', () => {
  let host: HTMLDivElement;
  let root: Root;

  const renderSettings = async () => {
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkspaceSettings />
        </ChakraProvider>
      )
    );
  };

  const renderTopBars = async () => {
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <LaunchpadTopBar />
          <AppMenu />
        </ChakraProvider>
      )
    );
  };

  const getVacuumButton = () =>
    Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) =>
      button.textContent?.includes('settings.databaseMaintenance.compactDatabase')
    );

  const openConfirmation = async () => {
    const button = getVacuumButton();
    expect(button).toBeDefined();
    await act(() => userEvent.click(button!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).not.toBeNull();
  };

  const getConfirmButton = () =>
    Array.from(document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')).find((button) =>
      button.textContent?.includes('settings.databaseMaintenance.compactDatabase')
    );

  beforeEach(() => {
    mocks.canManageAppConfig = true;
    mocks.apiFetch.mockReset().mockResolvedValue(new Response(null, { status: 204 }));
    mocks.notifyError.mockReset();
    mocks.notifySuccess.mockReset();
    accountLifecycle.activate('database-maintenance-settings-test', ':user:database-maintenance-settings-test');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('shows one Compact database button to admins and hides it from non-admins', async () => {
    await renderSettings();
    expect(getVacuumButton()).toBeDefined();

    mocks.canManageAppConfig = false;
    await renderSettings();
    expect(getVacuumButton()).toBeUndefined();
    expect(host.textContent).toContain('Clear saved data');
  });

  it('keeps maintenance out of both top-bar menus', async () => {
    await renderTopBars();
    expect(document.querySelector('[aria-label="settings.databaseMaintenance.menuLabel"]')).toBeNull();

    await act(() => userEvent.click(document.querySelector<HTMLElement>('[aria-label="topbar.appMenu.open"]')!));
    await expect.poll(() => document.querySelector('[role="menu"][data-state="open"]')).not.toBeNull();
    const items = Array.from(document.querySelectorAll('[role="menuitem"]'));
    expect(items.some((item) => item.textContent?.includes('settings.databaseMaintenance.menuLabel'))).toBe(false);
    expect(document.querySelector('[role="menuitem"][data-value="run-vacuum"]')).toBeNull();
  });

  it('requires confirmation and sends no request when canceled', async () => {
    await renderSettings();
    await openConfirmation();

    await act(() => userEvent.click(document.querySelector<HTMLElement>('[role="alertdialog"] button')!));

    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(mocks.apiFetch).not.toHaveBeenCalled();
  });

  it('posts once, disables confirmation while pending, and reports completion', async () => {
    let resolveRequest!: (response: Response) => void;
    const pendingRequest = new Promise<Response>((resolve) => {
      resolveRequest = resolve;
    });
    mocks.apiFetch.mockReturnValueOnce(pendingRequest);
    await renderSettings();
    await openConfirmation();

    const confirmButton = getConfirmButton();
    expect(confirmButton).toBeDefined();
    await act(() => userEvent.click(confirmButton!));

    expect(mocks.apiFetch).toHaveBeenCalledTimes(1);
    expect(mocks.apiFetch).toHaveBeenCalledWith('/api/v1/app/database/vacuum', {
      method: 'POST',
      signal: expect.any(AbortSignal),
    });
    await expect.poll(() => confirmButton!.disabled).toBe(true);
    expect(mocks.notifySuccess).not.toHaveBeenCalled();

    await act(() => resolveRequest(new Response(null, { status: 204 })));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(mocks.notifySuccess).toHaveBeenCalledWith('settings.databaseMaintenance.completed');
  });

  it('reports failures and permits retry', async () => {
    mocks.apiFetch.mockRejectedValueOnce(new Error('Database vacuum failed'));
    await renderSettings();
    await openConfirmation();
    await act(() => userEvent.click(getConfirmButton()!));

    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(mocks.notifyError).toHaveBeenCalledWith('settings.databaseMaintenance.failed', 'Database vacuum failed');

    await openConfirmation();
    await act(() => userEvent.click(getConfirmButton()!));

    await expect.poll(() => mocks.apiFetch).toHaveBeenCalledTimes(2);
    expect(mocks.notifySuccess).toHaveBeenCalledWith('settings.databaseMaintenance.completed');
  });
});
