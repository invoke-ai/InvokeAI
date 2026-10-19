import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const MANIFEST_PLACEHOLDER = 'self.__SW_MANIFEST__';

/**
 * Inject emitted asset names and their content-derived version into sw.js; identical output must produce the same
 * version.
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

    // Require one placeholder so injection cannot silently miss or replace a stray occurrence.
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
