import { afterEach, describe, expect, it, vi } from 'vitest';

import { retireDefaultServiceWorker } from './retireDefaultServiceWorker';

afterEach(() => vi.unstubAllGlobals());

describe('legacy frontend service-worker handoff', () => {
  const base = 'https://example.com/invoke';

  const setup = (scriptURL: string, controlled: boolean) => {
    const unregister = vi.fn(() => Promise.resolve(true));
    const getRegistration = vi.fn(() => Promise.resolve({ active: { scriptURL }, unregister }));
    const reload = vi.fn();
    const entries = new Map([
      ['invokeai-shell-old', new Set([`${base}/index.html`, 'https://example.com/other/index.html'])],
      ['invokeai-assets', new Set([`${base}/assets/main.js`, 'https://example.com/other/assets/main.js'])],
      ['invokeai-runtime', new Set([`${base}/locales/en.json`, 'https://example.com/other/locales/en.json'])],
      ['another-app', new Set([`${base}/index.html`])],
    ]);
    const cacheStorage = {
      keys: () => Promise.resolve([...entries.keys()]),
      open: (name: string) =>
        Promise.resolve({
          keys: () => Promise.resolve([...entries.get(name)!].map((url) => new Request(url))),
          delete: (request: Request) => Promise.resolve(entries.get(name)!.delete(request.url)),
        }),
    };
    vi.stubGlobal('navigator', { serviceWorker: { getRegistration, controller: controlled ? { scriptURL } : null } });
    vi.stubGlobal('window', { caches: cacheStorage, location: { reload } });
    vi.stubGlobal('caches', cacheStorage);
    return { unregister, getRegistration, reload, entries };
  };

  it('releases the deployment worker and only disposable Invoke caches before reloading', async () => {
    const { unregister, getRegistration, reload, entries } = setup(`${base}/sw.js`, true);
    expect(await retireDefaultServiceWorker(base)).toBe(true);
    expect(getRegistration).toHaveBeenCalledWith(`${base}/`);
    expect(unregister).toHaveBeenCalledOnce();
    expect(entries).toEqual(
      new Map([
        ['invokeai-shell-old', new Set(['https://example.com/other/index.html'])],
        ['invokeai-assets', new Set(['https://example.com/other/assets/main.js'])],
        ['invokeai-runtime', new Set(['https://example.com/other/locales/en.json'])],
        ['another-app', new Set([`${base}/index.html`])],
      ])
    );
    expect(reload).toHaveBeenCalledOnce();
  });

  it('continues boot without a reload when the document is already uncontrolled', async () => {
    const { reload } = setup(`${base}/sw.js`, false);
    expect(await retireDefaultServiceWorker(base)).toBe(false);
    expect(reload).not.toHaveBeenCalled();
  });

  it('leaves a different application worker and its caches alone', async () => {
    const { unregister, reload, entries } = setup('https://example.com/sw.js', true);
    expect(await retireDefaultServiceWorker(base)).toBe(false);
    expect(unregister).not.toHaveBeenCalled();
    expect(reload).not.toHaveBeenCalled();
    expect(entries.get('invokeai-assets')).toEqual(
      new Set([`${base}/assets/main.js`, 'https://example.com/other/assets/main.js'])
    );
  });

  it('boots normally when service workers are unavailable', async () => {
    vi.stubGlobal('navigator', {});
    expect(await retireDefaultServiceWorker(base)).toBe(false);
  });
});
