import { ChakraProvider } from '@chakra-ui/react';
import { apiFetchJson } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { HelpMenu } from './HelpMenu';

vi.mock('@platform/transport/http', () => ({ apiFetchJson: vi.fn() }));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { donateToInvokeAI: 'Donate to InvokeAI' },
        launchpad: { help: { community: 'Community', github: 'GitHub', label: 'Help' } },
      },
    },
  },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe.each([false, true])('HelpMenu donation link (compact=%s)', (compact) => {
  let host: HTMLDivElement;
  let root: Root;
  let client: QueryClient;

  beforeEach(async () => {
    vi.mocked(apiFetchJson).mockReset();
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <QueryClientProvider client={client}>
          <ChakraProvider value={system}>
            <I18nextProvider i18n={i18n}>
              <HelpMenu compact={compact} />
            </I18nextProvider>
          </ChakraProvider>
        </QueryClientProvider>
      )
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    client.clear();
    host.remove();
  });

  const trigger = () => page.getByRole('button', { name: 'Help' });

  it('resolves the setting on trigger hover so the link is already in the Community group when the menu opens', async () => {
    vi.mocked(apiFetchJson).mockResolvedValue({ show_donation_link: true });
    expect(apiFetchJson).not.toHaveBeenCalled();

    await userEvent.hover(trigger());
    await vi.waitFor(() => expect(client.getQueryState(['frontend-config'])?.status).toBe('success'));
    await trigger().click();

    const community = page.getByRole('group', { name: 'Community' });
    await expect.element(community.getByRole('menuitem', { name: 'Donate to InvokeAI' })).toBeVisible();
    const items = community.getByRole('menuitem').elements();
    expect(items.at(-1)?.textContent).toBe('Donate to InvokeAI');
    expect(apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('retries a failed request when the trigger regains focus', async () => {
    vi.mocked(apiFetchJson).mockRejectedValue(new Error('Server restarting'));

    await trigger().click();
    await vi.waitFor(() => expect(client.getQueryState(['frontend-config'])?.status).toBe('error'));
    await expect.element(page.getByRole('menuitem', { name: 'GitHub' })).toBeVisible();
    expect(page.getByRole('menuitem', { name: 'Donate to InvokeAI' }).query()).toBeNull();

    // The open menu keeps the item mounted, so recovery depends on the trigger asking again.
    vi.mocked(apiFetchJson).mockResolvedValue({ show_donation_link: true });
    await userEvent.keyboard('{Escape}');
    await expect.element(trigger()).toHaveFocus();
    await trigger().click();

    await expect.element(page.getByRole('menuitem', { name: 'Donate to InvokeAI' })).toBeVisible();
  });

  it('loads the link when assistive technology opens the menu without hovering or focusing the trigger', async () => {
    vi.mocked(apiFetchJson).mockResolvedValue({ show_donation_link: true });

    // HTMLElement.click() dispatches a bare click: no pointer events and no focus change.
    await act(() => (trigger().element() as HTMLElement).click());

    await expect.element(page.getByRole('menuitem', { name: 'Donate to InvokeAI' })).toBeVisible();
  });
});
