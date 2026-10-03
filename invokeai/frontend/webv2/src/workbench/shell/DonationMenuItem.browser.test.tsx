import { ChakraProvider, Menu } from '@chakra-ui/react';
import { apiFetchJson } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { DonationMenuItem } from './DonationMenuItem';

vi.mock('@platform/transport/http', () => ({ apiFetchJson: vi.fn() }));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  resources: { en: { translation: { common: { donateToInvokeAI: 'Donate to InvokeAI' } } } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('DonationMenuItem', () => {
  let host: HTMLDivElement;
  let root: Root;
  let client: QueryClient;

  beforeEach(() => {
    vi.mocked(apiFetchJson).mockReset();
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    client.clear();
    host.remove();
  });

  const render = () =>
    act(() =>
      root.render(
        <QueryClientProvider client={client}>
          <ChakraProvider value={system}>
            <I18nextProvider i18n={i18n}>
              <Menu.Root>
                <Menu.Trigger>Open menu</Menu.Trigger>
                <Menu.Positioner>
                  <Menu.Content>
                    <DonationMenuItem />
                    <Menu.Item value="other">Other action</Menu.Item>
                  </Menu.Content>
                </Menu.Positioner>
              </Menu.Root>
            </I18nextProvider>
          </ChakraProvider>
        </QueryClientProvider>
      )
    );

  it('offers a keyboard-selectable Sponsors link in a new tab when enabled', async () => {
    vi.mocked(apiFetchJson).mockResolvedValue({ show_donation_link: true });
    await render();
    await page.getByRole('button', { name: 'Open menu' }).click();
    const link = page.getByRole('menuitem', { name: 'Donate to InvokeAI' });
    await expect.element(link).toBeVisible();
    await expect.element(link).toHaveAttribute('href', 'https://github.com/sponsors/invoke-ai');
    await expect.element(link).toHaveAttribute('target', '_blank');
    await expect.element(link).toHaveAttribute('rel', 'noreferrer');
    await userEvent.keyboard('{Home}');
    await expect.element(page.getByRole('menu')).toHaveAttribute('aria-activedescendant', link.element().id);
    await expect.element(link).toHaveAttribute('data-highlighted');
    await userEvent.keyboard('{Escape}');
    await expect.element(page.getByRole('button', { name: 'Open menu' })).toHaveFocus();
  });

  it.each(['disabled', 'pending', 'failed'] as const)(
    'keeps the link hidden when configuration is %s',
    async (state) => {
      if (state === 'disabled') {
        vi.mocked(apiFetchJson).mockResolvedValue({ show_donation_link: false });
      } else if (state === 'failed') {
        vi.mocked(apiFetchJson).mockRejectedValue(new Error('Unavailable'));
      } else {
        vi.mocked(apiFetchJson).mockImplementation(() => new Promise(() => {}));
      }
      await render();
      await page.getByRole('button', { name: 'Open menu' }).click();
      await expect.element(page.getByRole('menuitem', { name: 'Other action' })).toBeVisible();
      if (state !== 'pending') {
        await vi.waitFor(() =>
          expect(client.getQueryState(['frontend-config'])?.status).toBe(state === 'failed' ? 'error' : 'success')
        );
      }
      expect(document.querySelector('a[href="https://github.com/sponsors/invoke-ai"]')).toBeNull();
    }
  );
});
