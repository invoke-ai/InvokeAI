import type * as SessionModule from '@features/identity/session';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const auth = vi.hoisted(() => ({ multiuser: true }));
vi.mock('@features/identity/session', async (importOriginal) => {
  const original = await importOriginal<typeof SessionModule>();
  const signedIn: SessionModule.AuthSession = {
    accountEpoch: 1,
    expiredUserId: null,
    multiuserEnabled: true,
    phase: 'ready',
    sessionExpired: false,
    setupRequired: false,
    strictPasswordChecking: false,
    switchingAccount: false,
    user: {
      created_at: '2026-01-01T00:00:00Z',
      display_name: 'Ada',
      email: 'ada@example.com',
      is_active: true,
      is_admin: true,
      last_login_at: null,
      updated_at: '2026-01-01T00:00:00Z',
      user_id: 'ada',
    },
  };

  const singleUser: SessionModule.AuthSession = { ...signedIn, multiuserEnabled: false, user: null };

  return { ...original, useAuthSession: () => (auth.multiuser ? signedIn : singleUser) };
});
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => null,
  useActiveProjectSelector: (selector: (project: { queue: { items: never[] } }) => unknown) =>
    selector({ queue: { items: [] } }),
  useWorkbenchCommands: () => ({}),
  useWorkbenchQueries: () => ({}),
}));
vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => vi.fn(),
}));
vi.mock('@workbench/workspace/useWorkspaceOpen', () => ({ useWorkspaceOpen: () => vi.fn() }));
vi.mock('@workbench/hotkeys/useShortcutBinding', () => ({
  useShortcutBinding: (commandId: string) =>
    commandId === 'app.openSettings' ? { aria: 'Control+Comma', display: 'Ctrl+,', parts: ['ctrl', ','] } : null,
}));
const openSettings = vi.hoisted(() => vi.fn());
vi.mock('@workbench/settings/settingsDialogStore', () => ({ openWorkbenchSettings: openSettings }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import {
  closeCommandPalette,
  commandPaletteStore,
  getCommandPaletteReturnFocusElement,
} from '@workbench/palette/paletteStore';

import { AppMenu } from './AppMenu';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  closeCommandPalette();
  auth.multiuser = true;
  await act(() => root.unmount());
  host.remove();
});

const menuItems = () => [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')];
const itemNamed = (name: string) => menuItems().find((item) => item.textContent?.startsWith(name));

const openMenu = async () => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={new QueryClient()}>
          <AppMenu />
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
  await act(() => userEvent.click(host.querySelector<HTMLButtonElement>('[aria-label="topbar.appMenu.open"]')!));
  await vi.waitFor(() => expect(itemNamed('topbar.appMenu.launchpad')).toBeDefined());
};

describe('AppMenu', () => {
  it('leads with Launchpad, Settings and the command palette as full items, each opening what it names', async () => {
    await openMenu();

    expect(
      menuItems()
        .map((item) => item.dataset.value)
        .slice(0, 3)
    ).toEqual(['launchpad', 'settings', 'command-palette']);
    const settings = itemNamed('common.settings')!;
    expect(settings.querySelector('kbd')?.closest('[aria-hidden="true"]')).not.toBeNull();
    expect(settings.getAttribute('aria-keyshortcuts')).toBe('Control+Comma');
    expect(itemNamed('commandPalette.buttonLabel')?.querySelector('kbd')).toBeNull();
    expect(itemNamed('commandPalette.buttonLabel')?.hasAttribute('aria-keyshortcuts')).toBe(false);
    expect(menuItems().flatMap((item) => item.getAttribute('aria-label') ?? [])).toEqual([
      'whatsNew.whatsNewInInvoke',
      'topbar.appMenu.documentation',
      'Discord',
    ]);

    const trigger = host.querySelector<HTMLButtonElement>('[aria-label="topbar.appMenu.open"]')!;
    await act(() => userEvent.click(itemNamed('common.settings')!));
    expect(openSettings).toHaveBeenCalledExactlyOnceWith(undefined, trigger);
    expect(commandPaletteStore.getSnapshot().isOpen).toBe(false);
    await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).toBeNull());

    await act(() => userEvent.click(trigger));
    const palette = await vi.waitFor(() => {
      expect(itemNamed('commandPalette.buttonLabel')).toBeDefined();
      return itemNamed('commandPalette.buttonLabel')!;
    });
    await act(() => userEvent.click(palette));
    expect(commandPaletteStore.getSnapshot().isOpen).toBe(true);
    expect(getCommandPaletteReturnFocusElement()).toBe(trigger);
  });

  it.each([
    ['a signed-in admin', true],
    ['single-user mode', false],
  ])('shows no account block for %s', async (_mode, multiuser) => {
    auth.multiuser = multiuser;
    await openMenu();

    const values = menuItems().map((item) => item.dataset.value);
    expect(values.slice(values.indexOf('widgets') + 1)).toEqual([
      'customize-workspace',
      'whats-new',
      'documentation',
      'discord',
    ]);
    expect(values).not.toContain('sign-out');
    expect(values).not.toContain('users');
    expect(document.querySelector('[role="menu"]')?.textContent).not.toContain('ada@example.com');
  });
});
