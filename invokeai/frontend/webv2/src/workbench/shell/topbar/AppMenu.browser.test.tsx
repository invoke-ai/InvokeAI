import type * as SessionModule from '@features/identity/session';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

vi.mock('@features/identity/session', async (importOriginal) => {
  const original = await importOriginal<typeof SessionModule>();
  const session: SessionModule.AuthSession = {
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
      is_admin: false,
      last_login_at: null,
      updated_at: '2026-01-01T00:00:00Z',
      user_id: 'ada',
    },
  };

  return { ...original, useAuthSession: () => session };
});
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => null,
  useActiveProjectSelector: (selector: (project: { queue: { items: never[] } }) => unknown) =>
    selector({ queue: { items: [] } }),
}));
vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => vi.fn(),
}));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('@workbench/hotkeys/useShortcutBinding', () => ({ useShortcutLabel: () => null }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

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
  await act(() => root.unmount());
  host.remove();
});

const dialog = () => document.querySelector<HTMLElement>('[role="dialog"]');

describe('AppMenu', () => {
  it('keeps Account settings open after the menu unmounts, and returns focus to the menu button', async () => {
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <AppMenu />
          </QueryClientProvider>
        </ChakraProvider>
      )
    );
    const trigger = host.querySelector<HTMLButtonElement>('[aria-label="topbar.appMenu.open"]')!;

    await act(() => userEvent.click(trigger));
    const item = await vi.waitFor(() => {
      const found = [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find(
        (candidate) => candidate.textContent === 'auth.accountSettings'
      );
      expect(found).toBeDefined();
      return found!;
    });
    await act(() => userEvent.click(item));

    // The menu content unmounts once its exit animation ends; the dialog must outlive it.
    await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).toBeNull());
    expect(dialog()?.textContent).toContain('auth.account');

    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => expect(dialog()).toBeNull());
    expect(document.activeElement).toBe(trigger);
  });
});
