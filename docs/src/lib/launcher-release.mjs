// The latest Invoke Launcher release, for the download page's labels: version, date and file sizes. The download
// buttons use GitHub's `releases/latest/download/` URLs, so a stale fallback never breaks a download.

export const LAUNCHER_RELEASE_API = 'https://api.github.com/repos/invoke-ai/launcher/releases/latest';

/** Each platform's installer, by file name suffix; `.blockmap` and `.yml` update metadata never match. */
const ASSET_SUFFIXES = {
  windows: '.exe',
  macos: '-arm64.dmg',
  linux: '.AppImage',
};

/**
 * Picks each platform's installer from a GitHub release; a platform without one is `null`. Releases carry each
 * installer twice, as a versioned file and as a `latest` alias of the same size.
 *
 * @param {{ tag_name: string, published_at: string, html_url: string, assets: { name: string, size: number }[] }} json
 */
export function parseLauncherRelease(json) {
  const assets = {};
  for (const [platform, suffix] of Object.entries(ASSET_SUFFIXES)) {
    const asset = json.assets.find((candidate) => candidate.name.endsWith(suffix));
    assets[platform] = asset ? { size: asset.size } : null;
  }
  return {
    version: json.tag_name.replace(/^v/, ''),
    publishedAt: json.published_at,
    notesUrl: json.html_url,
    assets,
  };
}

/** Bytes as the binary units Windows Explorer and GitHub show: `108 MB`, `1.2 GB`. */
export function formatSize(bytes) {
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 ? Math.round(value) : Math.round(value * 10) / 10} ${units[unit]}`;
}

/**
 * Fetches and parses the latest release. A token lifts GitHub's 60 requests/hour unauthenticated limit, which CI
 * runners share per IP. Throws on an HTTP error (a rate limit included) or after 5 seconds without an answer.
 */
export async function fetchLauncherRelease({ token, fetch = globalThis.fetch } = {}) {
  const headers = { Accept: 'application/vnd.github+json' };
  if (token) headers.Authorization = `Bearer ${token}`;
  // A stalled connection would otherwise hold up the build, or a dev render, until the socket gives up.
  const response = await fetch(LAUNCHER_RELEASE_API, { headers, signal: AbortSignal.timeout(5000) });
  if (!response.ok) throw new Error(`GitHub answered ${response.status} ${response.statusText}`);
  return parseLauncherRelease(await response.json());
}

let latest;

/**
 * The latest release, fetched once per build (or dev server), or `fallback` with one warning when the fetch fails.
 * The fallback is `src/data/launcher-release.json`; `pnpm run refresh-launcher-release` rewrites it from GitHub.
 */
export function loadLauncherRelease({ fallback, token, fetch = globalThis.fetch }) {
  latest ??= fetchLauncherRelease({ token, fetch }).catch((error) => {
    console.warn(
      `[download] Could not fetch the latest launcher release (${error.message}); using cached v${fallback.version}.`,
    );
    return fallback;
  });
  return latest;
}
