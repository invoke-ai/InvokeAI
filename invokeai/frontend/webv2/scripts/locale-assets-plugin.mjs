import { readFile, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const splitLocale = (translations) => {
  const { fonts, ...translation } = translations;
  return { translation, fonts: fonts ? { fonts } : {} };
};

/** Keep source locales readable; load font-library copy only when that UI opens. */
export const localeAssetsPlugin = ({ projectRoot }) => ({
  name: 'invokeai-locale-assets',
  configureServer(server) {
    server.middlewares.use(async (request, response, next) => {
      const match = request.url?.match(/\/locales\/([a-zA-Z0-9_-]+)(\.fonts)?\.json(?:\?.*)?$/);
      if (!match) {
        next();
        return;
      }
      try {
        const source = await readFile(resolve(projectRoot, `public/locales/${match[1]}.json`), 'utf8');
        const parts = splitLocale(JSON.parse(source));
        response.setHeader('Content-Type', 'application/json');
        response.end(JSON.stringify(match[2] ? parts.fonts : parts.translation));
      } catch (error) {
        if (error.code === 'ENOENT') {
          next();
        } else {
          next(error);
        }
      }
    });
  },
  async generateBundle() {
    const directory = resolve(projectRoot, 'public/locales');
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith('.json')) {
        continue;
      }
      const source = await readFile(resolve(directory, entry.name), 'utf8');
      let parsed;
      try {
        parsed = JSON.parse(source);
      } catch (error) {
        throw new Error(`Invalid locale JSON: ${entry.name}`, { cause: error });
      }
      const parts = splitLocale(parsed);
      // Vite copies public files before generateBundle; replace only the emitted copy.
      this.emitFile({ type: 'asset', fileName: `locales/${entry.name}`, source: JSON.stringify(parts.translation) });
      this.emitFile({
        type: 'asset',
        fileName: `locales/${entry.name.replace('.json', '.fonts.json')}`,
        source: JSON.stringify(parts.fonts),
      });
    }
  },
});
