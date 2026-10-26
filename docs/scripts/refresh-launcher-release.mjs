// Rewrites `src/data/launcher-release.json`, the download page's fallback when GitHub cannot be reached at build time,
// from the latest launcher release. Set GITHUB_TOKEN to avoid the unauthenticated rate limit.
import { writeFileSync } from 'node:fs';

import { fetchLauncherRelease } from '../src/lib/launcher-release.mjs';

const release = await fetchLauncherRelease({ token: process.env.GITHUB_TOKEN });
writeFileSync(new URL('../src/data/launcher-release.json', import.meta.url), `${JSON.stringify(release, null, 2)}\n`);
console.log(`Wrote launcher release v${release.version} (${release.publishedAt}).`);
