import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';

import { fetchLauncherRelease, formatSize, loadLauncherRelease, parseLauncherRelease } from './launcher-release.mjs';

// Trimmed from the v1.8.5 release: installers next to update metadata (`.blockmap`, `.yml`) and the macOS update zip.
const release = {
  tag_name: 'v1.8.5',
  published_at: '2026-10-10T21:05:09Z',
  html_url: 'https://github.com/invoke-ai/launcher/releases/tag/v1.8.5',
  assets: [
    { name: 'Invoke-Community-Edition-Setup-1.8.5.exe.blockmap', size: 118432 },
    { name: 'Invoke-Community-Edition-Setup-1.8.5.exe', size: 113202136 },
    { name: 'Invoke-Community-Edition-1.8.5-arm64-mac.zip', size: 128353194 },
    { name: 'Invoke-Community-Edition-1.8.5-arm64.dmg.blockmap', size: 139373 },
    { name: 'Invoke-Community-Edition-1.8.5-arm64.dmg', size: 132949779 },
    { name: 'Invoke.Community.Edition-latest.AppImage', size: 145780235 },
    { name: 'latest-linux.yml', size: 400 },
  ],
};

const respond = (body) => async () => ({ ok: true, json: async () => body });

/** The value's structure with each leaf replaced by its type, to compare data shapes. */
const shape = (value) =>
  value && typeof value === 'object'
    ? Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, shape(entry)]))
    : typeof value;

describe('parseLauncherRelease', () => {
  it('picks each platform installer, skipping update metadata', () => {
    assert.deepEqual(parseLauncherRelease(release), {
      version: '1.8.5',
      publishedAt: '2026-10-10T21:05:09Z',
      notesUrl: 'https://github.com/invoke-ai/launcher/releases/tag/v1.8.5',
      assets: { windows: { size: 113202136 }, macos: { size: 132949779 }, linux: { size: 145780235 } },
    });
  });

  it('leaves a platform without an installer null', () => {
    const withoutMac = { ...release, assets: release.assets.filter((a) => !a.name.includes('arm64')) };
    const parsed = parseLauncherRelease(withoutMac);
    assert.equal(parsed.assets.macos, null);
    assert.deepEqual(parsed.assets.windows, { size: 113202136 });
  });

  it('matches the shape of the cached fallback the page reads', () => {
    const cached = JSON.parse(readFileSync(new URL('../data/launcher-release.json', import.meta.url), 'utf8'));
    assert.deepEqual(shape(cached), shape(parseLauncherRelease(release)));
  });
});

describe('formatSize', () => {
  it('uses binary units, whole numbers from 10 up and one decimal below', () => {
    assert.equal(formatSize(113202136), '108 MB');
    assert.equal(formatSize(118432), '116 KB');
    assert.equal(formatSize(5 * 1024 * 1024 + 300 * 1024), '5.3 MB');
    assert.equal(formatSize(1.25 * 1024 ** 3), '1.3 GB');
  });
});

describe('fetchLauncherRelease', () => {
  it('sends the token as a bearer header, with a timeout, and parses the response', async () => {
    let request;
    const fetch = async (url, init) => {
      request = { url, init };
      return { ok: true, json: async () => release };
    };
    const parsed = await fetchLauncherRelease({ token: 'secret', fetch });
    assert.equal(request.url, 'https://api.github.com/repos/invoke-ai/launcher/releases/latest');
    assert.equal(request.init.headers.Authorization, 'Bearer secret');
    assert.ok(request.init.signal instanceof AbortSignal);
    assert.equal(parsed.version, '1.8.5');
  });

  it('rejects on a rate limit so the caller can fall back', async () => {
    const fetch = async () => ({ ok: false, status: 403, statusText: 'rate limit exceeded' });
    await assert.rejects(fetchLauncherRelease({ fetch }), /403/);
  });
});

describe('loadLauncherRelease', () => {
  // The release is loaded once per process, so this one test covers the fallback and the memoisation.
  it('falls back with one warning on a failed fetch, and keeps that result', async (t) => {
    const warn = t.mock.method(console, 'warn', () => {});
    const fallback = { version: '1.0.0' };
    const failing = async () => {
      throw new Error('The operation was aborted due to timeout');
    };

    assert.equal(await loadLauncherRelease({ fallback, fetch: failing }), fallback);
    assert.equal(warn.mock.callCount(), 1);
    assert.match(warn.mock.calls[0].arguments[0], /timeout.*cached v1\.0\.0/);

    const fetch = t.mock.fn(respond(release));
    assert.equal(await loadLauncherRelease({ fallback, fetch }), fallback);
    assert.equal(fetch.mock.callCount(), 0);
    assert.equal(warn.mock.callCount(), 1);
  });
});
