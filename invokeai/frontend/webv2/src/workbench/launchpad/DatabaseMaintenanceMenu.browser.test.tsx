import type * as IdentityModule from '@features/identity';
import type * as HttpModule from '@platform/transport/http';

/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

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
vi.mock('@tanstack/react-router', () => ({ useNavigate: () => vi.fn() }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => null,
  useActiveProjectSelector: (selector: (project: { queue: { items: never[] } }) => unknown) =>
    selector({ queue: { items: [] } }),
}));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('@workbench/useNotify', () => ({
  useNotify: () => ({ error: mocks.notifyError, success: mocks.notifySuccess }),
}));
vi.mock('./useTopbarShortcut', () => ({ useTopbarShortcut: () => null }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { DatabaseMaintenanceMenu } from '@workbench/launchpad/LaunchpadTopBar';
import { AppMenu } from '@workbench/shell/topbar/AppMenu';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('database maintenance menu', () => {
  let host: HTMLDivElement;
  let root: Root;

  const renderMenu = async (variant: 'launchpad' | 'app-menu' = 'launchpad') => {
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          {variant === 'app-menu' ? <AppMenu /> : <DatabaseMaintenanceMenu />}
        </ChakraProvider>
      )
    );
  };

  const openLaunchpadMenu = async () => {
    const trigger = document.querySelector<HTMLElement>('[aria-label="settings.databaseMaintenance.menuLabel"]');
    expect(trigger).not.toBeNull();
    await act(() => userEvent.click(trigger!));
    await expect.poll(() => document.querySelector('[role="menu"][data-state="open"]')).not.toBeNull();
  };

  const openAppMenuVacuumAction = async () => {
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[aria-label="topbar.appMenu.open"]')!));
    await expect.poll(() => document.querySelector('[role="menu"][data-state="open"]')).not.toBeNull();
    const submenu = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find((item) =>
      item.textContent?.includes('settings.databaseMaintenance.menuLabel')
    );
    expect(submenu).toBeDefined();
    await act(() => userEvent.click(submenu!));
    await expect.poll(() => document.querySelector('[role="menuitem"][data-value="run-vacuum"]')).not.toBeNull();
  };

  const openConfirmation = async () => {
    const item = document.querySelector<HTMLElement>('[role="menuitem"][data-value="run-vacuum"]');
    expect(item).not.toBeNull();
    await act(() => userEvent.click(item!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).not.toBeNull();
  };

  const getConfirmButton = () =>
    Array.from(document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')).find((button) =>
      button.textContent?.includes('settings.databaseMaintenance.runVacuum')
    );

  beforeEach(async () => {
    mocks.canManageAppConfig = true;
    mocks.apiFetch.mockReset().mockResolvedValue(new Response(null, { status: 204 }));
    mocks.notifyError.mockReset();
    mocks.notifySuccess.mockReset();
    accountLifecycle.activate('database-maintenance-test', ':user:database-maintenance-test');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await renderMenu();
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('hides maintenance controls when the session cannot manage app configuration', async () => {
    mocks.canManageAppConfig = false;
    await renderMenu();

    expect(document.querySelector('[aria-label="settings.databaseMaintenance.menuLabel"]')).toBeNull();
    await renderMenu('app-menu');
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[aria-label="topbar.appMenu.open"]')!));
    await expect.poll(() => document.querySelector('[role="menu"][data-state="open"]')).not.toBeNull();
    expect(
      Array.from(document.querySelectorAll('[role="menuitem"]')).some((item) =>
        item.textContent?.includes('settings.databaseMaintenance.menuLabel')
      )
    ).toBe(false);
  });

  it('shows only the VACUUM action in the launchpad menu', async () => {
    await openLaunchpadMenu();

    const items = Array.from(document.querySelectorAll('[role="menuitem"]'));
    expect(items).toHaveLength(1);
    expect(items[0]?.textContent).toContain('settings.databaseMaintenance.runVacuum');
  });

  it('adds the single VACUUM action to the AppMenu Manage submenu', async () => {
    await renderMenu('app-menu');
    await openAppMenuVacuumAction();
    expect(document.querySelectorAll('[role="menuitem"][data-value="run-vacuum"]')).toHaveLength(1);
  });

  it('requires confirmation and sends no request when canceled', async () => {
    await openLaunchpadMenu();
    await openConfirmation();

    await act(() => userEvent.click(document.querySelector<HTMLElement>('[role="alertdialog"] button')!));

    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(mocks.apiFetch).not.toHaveBeenCalled();
  });

  it('sends one POST, disables confirmation while pending, and reports completion after 204', async () => {
    let resolveRequest!: (response: Response) => void;
    const pendingRequest = new Promise<Response>((resolve) => {
      resolveRequest = resolve;
    });
    mocks.apiFetch.mockReturnValueOnce(pendingRequest);
    await openLaunchpadMenu();
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

  it('supports keyboard activation and Escape dismissal of the confirmation', async () => {
    const trigger = document.querySelector<HTMLElement>('[aria-label="settings.databaseMaintenance.menuLabel"]');
    expect(trigger).not.toBeNull();
    await act(async () => {
      trigger!.focus();
      await userEvent.keyboard('{Enter}');
    });
    await expect.poll(() => document.querySelector('[role="menu"][data-state="open"]')).not.toBeNull();

    const item = document.querySelector<HTMLElement>('[role="menuitem"][data-value="run-vacuum"]');
    expect(item).not.toBeNull();
    await act(async () => {
      item!.focus();
      await userEvent.keyboard('{Enter}');
    });
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).not.toBeNull();
    await act(() => userEvent.keyboard('{Escape}'));

    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(mocks.apiFetch).not.toHaveBeenCalled();
  });

  it('reports failures and permits retry', async () => {
    mocks.apiFetch.mockRejectedValueOnce(new Error('Database vacuum failed'));
    await openLaunchpadMenu();
    await openConfirmation();
    await act(() => userEvent.click(getConfirmButton()!));

    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(mocks.notifyError).toHaveBeenCalledWith('settings.databaseMaintenance.failed', 'Database vacuum failed');

    await openLaunchpadMenu();
    await openConfirmation();
    await act(() => userEvent.click(getConfirmButton()!));

    await expect.poll(() => mocks.apiFetch).toHaveBeenCalledTimes(2);
    expect(mocks.notifySuccess).toHaveBeenCalledWith('settings.databaseMaintenance.completed');
  });
});
