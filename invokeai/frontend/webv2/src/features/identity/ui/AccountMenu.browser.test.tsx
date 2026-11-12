import type * as SessionModule from '@features/identity/session';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const account = vi.hoisted(() => ({ displayName: 'Ada' as string | null }));
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

  return {
    ...original,
    useAuthSession: () => ({ ...session, user: { ...session.user!, display_name: account.displayName } }),
  };
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
  account.displayName = 'Ada';
  await act(() => root.unmount());
  host.remove();
});

const dialog = () => document.querySelector<HTMLElement>('[role="dialog"]');

const renderMenu = (compact = false) =>
  act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={new QueryClient()}>
          <AccountMenu compact={compact} />
        </QueryClientProvider>
      </ChakraProvider>
    )
  );

/** How far the initials' ink centre sits from the avatar's centre, from the font's own glyph metrics. */
const initialsOffset = () => {
  const avatar = host.querySelector('[data-scope="avatar"][data-part="root"]')!;
  const initials = avatar.querySelector<HTMLElement>('[data-part="fallback"]')!;
  const style = getComputedStyle(initials);
  const context = document.createElement('canvas').getContext('2d')!;
  context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  const ink = context.measureText(initials.textContent!.toUpperCase());
  // A zero-height inline block rests on the baseline.
  const marker = document.createElement('span');
  marker.style.cssText = 'display: inline-block; height: 0; vertical-align: baseline; width: 0';
  initials.append(marker);
  const baseline = marker.getBoundingClientRect().top;
  marker.remove();
  const range = document.createRange();
  range.selectNodeContents(initials);
  const start = range.getBoundingClientRect().left;
  const circle = avatar.getBoundingClientRect();

  return {
    x: start + (ink.actualBoundingBoxRight - ink.actualBoundingBoxLeft) / 2 - (circle.left + circle.width / 2),
    y: baseline + (ink.actualBoundingBoxDescent - ink.actualBoundingBoxAscent) / 2 - (circle.top + circle.height / 2),
  };
};

describe('AccountMenu', () => {
  // Untrimmed, the font's descender room lifts the capitals half a pixel at this size; trimmed, they sit dead centre.
  it.each([
    ['row', false],
    ['icon-only rail', true],
  ])('centres the initials in the avatar in the %s', async (_width, compact) => {
    await renderMenu(compact);
    await document.fonts.ready;

    const offset = initialsOffset();
    expect(Math.abs(offset.x)).toBeLessThanOrEqual(0.25);
    expect(Math.abs(offset.y)).toBeLessThanOrEqual(0.25);
  });

  it('shows the name over the email, or the role for an account known only by its email', async () => {
    await renderMenu();
    const trigger = () => host.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!;
    const lines = () => [...trigger().querySelectorAll('p')].map((line) => line.textContent);
    // Named by these lines (the avatar is decorative), so the spoken name is what is shown.
    expect(trigger().getAttribute('aria-label')).toBeNull();
    expect(trigger().querySelector('[data-scope="avatar"]')?.getAttribute('aria-hidden')).toBe('true');
    expect(lines()).toEqual(['Ada', 'ada@example.com']);

    account.displayName = null;
    await renderMenu();
    expect(lines()).toEqual(['ada@example.com', 'users.user']);
  });

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
