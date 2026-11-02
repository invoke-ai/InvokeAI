import type * as SessionModule from '@features/identity/session';

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
vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => vi.fn(),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { AccountMenu } from './AccountMenu';

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

describe('AccountMenu', () => {
  it('returns focus to the menu button when Account settings closes', async () => {
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <AccountMenu />
          </QueryClientProvider>
        </ChakraProvider>
      )
    );
    const trigger = host.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!;

    await act(() => userEvent.click(trigger));
    const item = await vi.waitFor(() => {
      const found = [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find(
        (candidate) => candidate.textContent === 'auth.accountSettings'
      );
      expect(found).toBeDefined();
      return found!;
    });
    await act(() => userEvent.click(item));
    await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).toBeNull());
    expect(dialog()?.textContent).toContain('auth.account');

    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => expect(dialog()).toBeNull());
    expect(document.activeElement).toBe(trigger);
  });
});
