import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import test from 'node:test';
import { build, createServer } from 'vite';

import { localeAssetsPlugin } from './locale-assets-plugin.mjs';

const createFixture = async (context) => {
  const projectRoot = await mkdtemp(resolve(tmpdir(), 'invokeai-locale-build-'));
  // Retried: Vite can still be flushing its `.vite` cache into the fixture as it is removed, and an `after` hook that
  // throws stops every hook registered after it from running.
  context.after(() => rm(projectRoot, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 }));
  await mkdir(resolve(projectRoot, 'public/locales'), { recursive: true });
  await writeFile(
    resolve(projectRoot, 'index.html'),
    '<html><body><script type="module" src="/main.js"></script></body></html>'
  );
  await writeFile(
    resolve(projectRoot, 'main.js'),
    'document.body.dataset.localeVersion = import.meta.env.LOCALE_VERSION;'
  );
  return projectRoot;
};

const buildFixture = (projectRoot) =>
  build({
    root: projectRoot,
    configFile: false,
    logLevel: 'silent',
    plugins: [localeAssetsPlugin({ projectRoot })],
  });

/** Reads the version the build injected into the bundled entry. */
const readBuiltLocaleVersion = async (projectRoot) => {
  const directory = resolve(projectRoot, 'dist/assets');
  const scripts = (await readdir(directory)).filter((name) => name.endsWith('.js'));
  const sources = await Promise.all(scripts.map((name) => readFile(resolve(directory, name), 'utf8')));
  const versions = sources.flatMap((source) =>
    [...source.matchAll(/localeVersion\s*=\s*(["'`])(.*?)\1/g)].map((match) => match[2])
  );
  assert.equal(versions.length, 1, 'The entry must assign one string literal as the locale version.');
  return versions[0];
};

test('production locale assets replace public copies without changing Unicode, values, or source files', async (context) => {
  const projectRoot = await createFixture(context);
  const translations = {
    settings: { title: '設定 🎨', description: 'Réglages — العربية', escaped: 'Line one\n"Line two"\\' },
    list: ['中文', '{{count}} settings', ''],
    enabled: true,
    count: 42,
    optional: null,
  };
  const formatted = `${JSON.stringify(translations, null, 2)}\n`;
  await writeFile(resolve(projectRoot, 'public/locales/en.json'), formatted);
  await writeFile(resolve(projectRoot, 'public/locales/ja.json'), formatted);
  await writeFile(resolve(projectRoot, 'public/keep.txt'), '  Keep public whitespace.\n');

  await buildFixture(projectRoot);

  for (const language of ['en', 'ja']) {
    const built = await readFile(resolve(projectRoot, `dist/locales/${language}.json`), 'utf8');
    assert.deepEqual(JSON.parse(built), translations);
    assert.equal(built, JSON.stringify(translations));
    assert.ok(Buffer.byteLength(built) < Buffer.byteLength(formatted));
    assert.equal(await readFile(resolve(projectRoot, `public/locales/${language}.json`), 'utf8'), formatted);
  }
  assert.equal(await readFile(resolve(projectRoot, 'dist/keep.txt'), 'utf8'), '  Keep public whitespace.\n');
});

test('the injected locale version follows the emitted locale files across builds and project paths', async (context) => {
  const french = { common: { retry: 'Réessayer' } };
  const writeLocales = async (projectRoot) => {
    await writeFile(resolve(projectRoot, 'public/locales/en.json'), JSON.stringify({ common: { retry: 'Retry' } }));
    await writeFile(resolve(projectRoot, 'public/locales/fr.json'), JSON.stringify(french));
  };
  const first = await createFixture(context);
  const second = await createFixture(context);
  await writeLocales(first);
  await writeLocales(second);

  await buildFixture(first);
  await buildFixture(second);
  const version = await readBuiltLocaleVersion(first);
  assert.ok(version && version !== 'undefined', `Expected a locale version, got ${JSON.stringify(version)}.`);
  assert.equal(await readBuiltLocaleVersion(second), version);

  // Reformatting a source leaves the emitted files, and so the version, unchanged.
  await writeFile(resolve(second, 'public/locales/fr.json'), `${JSON.stringify(french, null, 2)}\n`);
  await buildFixture(second);
  assert.equal(await readBuiltLocaleVersion(second), version);

  await writeFile(resolve(second, 'public/locales/fr.json'), JSON.stringify({ common: { retry: 'Encore' } }));
  await buildFixture(second);
  assert.notEqual(await readBuiltLocaleVersion(second), version);
});

test('invalid locale JSON fails the production build with the failing filename', async (context) => {
  const projectRoot = await createFixture(context);
  await writeFile(resolve(projectRoot, 'public/locales/broken.json'), '{ invalid');

  await assert.rejects(buildFixture(projectRoot), /Invalid locale JSON: broken\.json/);
});

test('font namespace loads separately in production and development, preserving fallback locales', async (context) => {
  const projectRoot = await createFixture(context);
  const translations = { common: { retry: 'Retry' }, fonts: { title: 'Fonts 🎨' } };
  await writeFile(resolve(projectRoot, 'public/locales/en.json'), JSON.stringify(translations));
  await writeFile(resolve(projectRoot, 'public/locales/fr.json'), JSON.stringify({ common: { retry: 'Réessayer' } }));
  await buildFixture(projectRoot);
  assert.deepEqual(JSON.parse(await readFile(resolve(projectRoot, 'dist/locales/en.json'), 'utf8')), {
    common: translations.common,
  });
  assert.deepEqual(JSON.parse(await readFile(resolve(projectRoot, 'dist/locales/en.fonts.json'), 'utf8')), {
    fonts: translations.fonts,
  });
  assert.deepEqual(JSON.parse(await readFile(resolve(projectRoot, 'dist/locales/fr.fonts.json'), 'utf8')), {});
  const server = await createServer({
    root: projectRoot,
    configFile: false,
    logLevel: 'silent',
    plugins: [localeAssetsPlugin({ projectRoot })],
    server: { port: 0, host: '127.0.0.1' },
  });
  // Closed here rather than in an `after` hook: hooks run in registration order, so the fixture's removal would run
  // first, race the server's cache writes, and — if it threw — leave the server open and the test process hanging.
  try {
    await server.listen();
    const origin = server.resolvedUrls.local[0];
    for (const filename of ['en.json', 'en.fonts.json', 'fr.fonts.json']) {
      // Versioned like the client's requests.
      const response = await fetch(new URL(`locales/${filename}?v=0123456789ab`, origin));
      assert.equal(response.status, 200);
      assert.deepEqual(
        await response.json(),
        JSON.parse(await readFile(resolve(projectRoot, `dist/locales/${filename}`), 'utf8'))
      );
    }
  } finally {
    await server.close();
  }
});
