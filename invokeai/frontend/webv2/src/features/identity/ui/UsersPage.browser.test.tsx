import type * as ApiModule from '@features/identity/data/api';
import type * as SessionModule from '@features/identity/session';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

const admin = vi.hoisted(() => ({
  created_at: '2026-01-01T00:00:00Z',
  display_name: 'Ada',
  email: 'ada@example.com',
  is_active: true,
  is_admin: true,
  last_login_at: null,
  updated_at: '2026-01-01T00:00:00Z',
  user_id: 'ada',
}));

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
    user: admin,
  };

  return { ...original, useAuthSession: () => session };
});
/** Each test answers the directory's loads in turn. */
const directory = vi.hoisted(() => ({ loads: [] as (() => Promise<unknown>)[] }));
vi.mock('@features/identity/data/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ApiModule>()),
  listUsers: () => (directory.loads.shift() ?? (() => Promise.resolve([admin])))(),
}));
// A stable `t`, as react-i18next's is: the directory reloads whenever it changes.
const translation = vi.hoisted(() => ({
  t: (key: string, values?: { count?: number }) => (values?.count === undefined ? key : `${key}:${values.count}`),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => translation }));

import { UsersPage } from './UsersPage';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  directory.loads = [];
  await act(() => root.unmount());
  host.remove();
});

const renderPage = () =>
  act(() =>
    root.render(
      <ChakraProvider value={system}>
        <UsersPage />
      </ChakraProvider>
    )
  );

describe('UsersPage', () => {
  it('names the page once, with the account count above the directory', async () => {
    await renderPage();

    await expect.element(page.getByText('users.accountCount:1')).toBeVisible();
    expect([...host.querySelectorAll('h1, h2, h3, h4, h5, h6')].map((heading) => heading.textContent)).toEqual([
      'users.title',
    ]);
    expect(host.textContent?.match(/users\.title/g)).toHaveLength(1);
    await expect.element(page.getByText('users.description')).toBeVisible();
    await expect.element(page.getByRole('button', { name: 'users.addUser' })).toBeVisible();
  });

  it('describes the page once while the directory loads, with no count yet', async () => {
    directory.loads.push(() => new Promise(() => {}));
    await renderPage();

    await expect.element(page.getByRole('button', { name: 'users.addUser' })).toBeVisible();
    expect(host.textContent?.match(/users\.description/g)).toHaveLength(1);
    expect(host.textContent).not.toContain('users.accountCount');
    expect(host.querySelector('table')).toBeNull();
  });

  it('offers a retry when the directory fails to load, then shows the count', async () => {
    directory.loads.push(() => Promise.reject(new Error('Backend offline')));
    await renderPage();

    await expect.element(page.getByText('Backend offline')).toBeVisible();
    expect(host.textContent).not.toContain('users.accountCount');
    await page.getByRole('button', { name: 'common.retry' }).click();

    await expect.element(page.getByText('users.accountCount:1')).toBeVisible();
    expect(host.textContent).not.toContain('Backend offline');
  });
});
