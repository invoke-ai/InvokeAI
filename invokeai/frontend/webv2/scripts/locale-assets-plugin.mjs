import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const catalogKeys = (value, prefix = '', result = new Set()) => {
  for (const [key, child] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (typeof child === 'string' && child.trim()) {
      result.add(path);
    } else if (child && typeof child === 'object') {
      catalogKeys(child, path, result);
    }
  }
  return result;
};

const splitLocale = (translations) => {
  const { fonts, ...translation } = translations;
  const { metadataLabels, ...preview } = translation.widgets?.preview ?? {};
  if (metadataLabels !== undefined) {
    translation.widgets = { ...translation.widgets, preview };
  }
  return {
    translation,
    fonts: fonts ? { fonts } : {},
    metadata: metadataLabels === undefined ? {} : { widgets: { preview: { metadataLabels } } },
  };
};

/** The files a build emits for `public/locales`, sorted by name so the version does not depend on readdir order. */
const buildLocaleAssets = async (projectRoot) => {
  const directory = resolve(projectRoot, 'public/locales');
  const entries = await readdir(directory, { withFileTypes: true });
  const names = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => entry.name)
    .sort();
  const sources = await Promise.all(names.map((name) => readFile(resolve(directory, name), 'utf8')));
  return names.flatMap((name, index) => {
    let parsed;
    try {
      parsed = JSON.parse(sources[index]);
    } catch (error) {
      throw new Error(`Invalid locale JSON: ${name}`, { cause: error });
    }
    const parts = splitLocale(parsed);
    return [
      { fileName: `locales/${name}`, source: JSON.stringify(parts.translation) },
      { fileName: `locales/${name.replace('.json', '.fonts.json')}`, source: JSON.stringify(parts.fonts) },
      { fileName: `locales/${name.replace('.json', '.metadata.json')}`, source: JSON.stringify(parts.metadata) },
    ];
  });
};

/**
 * Keep source locales readable; load font-library and metadata labels only when their UI opens. Locale URLs stay
 * the same across releases, so the injected content version tells the client and service worker which files a build carries.
 */
export const localeAssetsPlugin = ({ projectRoot }) => {
  let assets;
  let availability;

  return {
    name: 'invokeai-locale-assets',
    async config() {
      assets = await buildLocaleAssets(projectRoot);
      const catalogs = assets.filter((asset) => !asset.fileName.endsWith('.fonts.json'));
      const english = catalogKeys(JSON.parse(catalogs.find((asset) => asset.fileName === 'locales/en.json').source));
      availability = Object.fromEntries(
        catalogs.flatMap((asset) => {
          const keys = catalogKeys(JSON.parse(asset.source));
          const translated = [...keys].filter((key) => english.has(key)).length;
          const language = asset.fileName.slice('locales/'.length, -'.json'.length);
          // Coverage describes content; readiness is an explicit product declaration, never a percentage threshold.
          return translated ? [[language, { translated, ...(language === 'en' ? { ready: true } : {}) }]] : [];
        })
      );
      // Hash the emitted files, not the sources: formatting-only edits keep the version, transform changes do not.
      // JSON keeps the boundaries between file names and contents unambiguous.
      const version = createHash('sha256').update(JSON.stringify(assets)).digest('hex').slice(0, 12);
      // An env define, not a global: Vitest browser mode re-encodes string globals, wrapping them in quotes.
      return {
        define: {
          'import.meta.env.LOCALE_VERSION': JSON.stringify(version),
          'import.meta.env.LOCALE_AVAILABILITY': JSON.stringify(JSON.stringify(availability)),
        },
      };
    },
    transformIndexHtml(html) {
      const ready = Object.keys(availability).filter((language) => availability[language].ready);
      return html.replace('__READY_CONTENT_LANGUAGES__', JSON.stringify(ready));
    },
    configureServer(server) {
      // Read per request so edits show without a restart. The version stays at its startup value, which only the
      // production service worker relies on.
      server.middlewares.use(async (request, response, next) => {
        const match = request.url?.match(/\/locales\/([a-zA-Z0-9_-]+)(?:\.(fonts|metadata))?\.json(?:\?.*)?$/);
        if (!match) {
          next();
          return;
        }
        try {
          const source = await readFile(resolve(projectRoot, `public/locales/${match[1]}.json`), 'utf8');
          const parts = splitLocale(JSON.parse(source));
          response.setHeader('Content-Type', 'application/json');
          response.end(JSON.stringify(parts[match[2] ?? 'translation']));
        } catch (error) {
          if (error.code === 'ENOENT') {
            next();
          } else {
            next(error);
          }
        }
      });
    },
    generateBundle() {
      // Emit exactly the files config() hashed. Vite copies public files before generateBundle; replace only the
      // emitted copy.
      for (const { fileName, source } of assets) {
        this.emitFile({ type: 'asset', fileName, source });
      }
    },
  };
};
