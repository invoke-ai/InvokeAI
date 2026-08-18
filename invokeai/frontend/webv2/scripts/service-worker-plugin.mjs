import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const MANIFEST_PLACEHOLDER = 'self.__SW_MANIFEST__';

/**
 * Emits `sw.js` at the bundle root from the source worker in
 * `src/platform/pwa/sw.js`, injecting the build's asset manifest: the emitted
 * `assets/*` file names plus a version hash derived from them. The worker uses
 * the list for cache-first serving and stale-entry pruning; deriving the
 * version from the same list keeps rebuilds of identical output byte-stable.
 */
export const serviceWorkerPlugin = ({ projectRoot }) => ({
  apply: 'build',
  name: 'invokeai-service-worker',
  async generateBundle(_outputOptions, bundle) {
    const assets = Object.keys(bundle)
      .filter((fileName) => fileName.startsWith('assets/'))
      .sort();
    const version = createHash('sha256').update(assets.join('\n')).digest('hex').slice(0, 12);
    const source = await readFile(resolve(projectRoot, 'src/platform/pwa/sw.js'), 'utf8');
    const occurrences = source.split(MANIFEST_PLACEHOLDER).length - 1;

    // Exactly one: zero means the placeholder was renamed away, more than one
    // means a stray mention (e.g. in a comment) would swallow the injection.
    if (occurrences !== 1) {
      throw new Error(`Service worker source must contain ${MANIFEST_PLACEHOLDER} exactly once, found ${occurrences}.`);
    }

    this.emitFile({
      fileName: 'sw.js',
      source: source.replace(MANIFEST_PLACEHOLDER, JSON.stringify({ assets, version })),
      type: 'asset',
    });
  },
});
