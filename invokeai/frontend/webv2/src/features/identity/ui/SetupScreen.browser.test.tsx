import type { AuthSession } from '@features/identity/session';

import { ChakraProvider } from '@chakra-ui/react';
import { DOCS_URL } from '@platform/runtime/appMetadata';
import { createExternalStore } from '@platform/state/externalStore';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { SetupScreen } from './SetupScreen';

const sessionStore = createExternalStore<AuthSession>({
  accountEpoch: 0,
  expiredUserId: null,
  hasSingleUserWork: false,
  multiuserEnabled: true,
  phase: 'ready',
  sessionExpired: false,
  setupRequired: true,
  strictPasswordChecking: false,
  switchingAccount: false,
  user: null,
});

vi.mock('@features/identity/session', () => ({
  completeAdminSetup: vi.fn(),
  isLoginAttemptSupersededError: () => false,
  useAuthSession: () => sessionStore.useSnapshot(),
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <SetupScreen />
        </I18nextProvider>
      </ChakraProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  sessionStore.patchSnapshot({ hasSingleUserWork: false });
});

describe('setup screen', () => {
  it('explains where earlier work went only when the server reports some', async () => {
    const guide = page.getByRole('link', { name: i18n.t('auth.setupEarlierWorkGuide') });
    const submit = page.getByRole('button', { name: i18n.t('auth.createAdminAccount') });

    expect(page.getByRole('note').query()).toBeNull();
    await expect.element(submit).not.toHaveAttribute('aria-describedby');

    await act(() => sessionStore.patchSnapshot({ hasSingleUserWork: true }));

    await expect.element(page.getByRole('note').getByText(i18n.t('auth.setupEarlierWorkDescription'))).toBeVisible();
    await expect.element(submit).toHaveAccessibleDescription(i18n.t('auth.setupEarlierWorkDescription'));
    await expect
      .element(guide)
      .toHaveAttribute('href', `${DOCS_URL}configuration/multi-user-mode/admin-guide/#if-you-have-already-upgraded`);
    await expect.element(guide).toHaveAttribute('target', '_blank');
  });
});
