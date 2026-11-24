import type * as ModelsApi from '@features/models/data/api';

import { ChakraProvider } from '@chakra-ui/react';
import { setExternalProvidersSnapshotForTests } from '@features/models/data/externalProvidersStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ExternalProviderKeyCards } from './ExternalProviderKeyCards';

const { save } = vi.hoisted(() => ({ save: vi.fn() }));
vi.mock('@features/models/data/api', async (original) => ({
  ...(await original<typeof ModelsApi>()),
  setExternalProviderConfig: save,
}));
vi.mock('@features/models/data/installsStore', () => ({ refreshInstalls: () => undefined }));
vi.mock('@features/models/data/startersStore', () => ({ refreshStartersIfLoaded: () => undefined }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

it('preserves a custom image endpoint on key rotation and requires a fresh key to switch endpoints', async () => {
  accountLifecycle.activate('admin');
  const config = { api_key_configured: true, base_url: 'https://images.example.test', provider_id: 'minimax' };
  setExternalProvidersSnapshotForTests({ configs: [config], status: 'loaded' });
  save.mockResolvedValueOnce(config).mockResolvedValueOnce({ ...config, base_url: 'https://api.minimax.cn' });
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  try {
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <ExternalProviderKeyCards onError={vi.fn()} />
        </ChakraProvider>
      )
    );
    const region = host.querySelector('select')!;
    const password = host.querySelector<HTMLInputElement>('input[type="password"]')!;
    const button = [...host.querySelectorAll('button')].find((candidate) => candidate.textContent === 'common.save')!;
    expect(region.value).toBe('custom');
    await act(() => userEvent.fill(password, 'rotated'));
    await act(() => userEvent.click(button));
    expect(save.mock.calls[0]?.slice(0, 2)).toEqual([
      'minimax',
      { api_key: 'rotated', base_url: 'https://images.example.test' },
    ]);
    await act(() => userEvent.selectOptions(region, 'https://api.minimax.cn'));
    expect(button.disabled).toBe(true);
    await act(() => userEvent.fill(password, 'cn-key'));
    await act(() => userEvent.click(button));
    expect(save.mock.calls[1]?.slice(0, 2)).toEqual([
      'minimax',
      { api_key: 'cn-key', base_url: 'https://api.minimax.cn' },
    ]);
    await act(() => userEvent.selectOptions(region, 'custom'));
    expect(host.querySelector<HTMLInputElement>('[aria-label="models.baseUrlFor"]')?.value).toBe('');
    expect(button.disabled).toBe(true);
  } finally {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  }
});
