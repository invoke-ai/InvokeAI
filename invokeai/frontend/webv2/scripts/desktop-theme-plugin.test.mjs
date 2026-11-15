import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import process from 'node:process';
import test from 'node:test';
import { build, createServer } from 'vite';

import { desktopThemePlugin } from './desktop-theme-plugin.mjs';

const MARKER = '<meta name="invokeai-desktop-theme" content="" />';
const document = `<html><head>${MARKER}</head><body>Workbench</body></html>`;

const fixture = async (context) => {
  const root = await mkdtemp(resolve(tmpdir(), 'invokeai-desktop-theme-'));
  context.after(() => rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 }));
  const state = resolve(root, 'state');
  const palette = resolve(state, 'omarchy/current/theme/colors.toml');
  await mkdir(resolve(palette, '..'), { recursive: true });
  await writeFile(resolve(root, 'index.html'), document);
  const env = { ...process.env, XDG_STATE_HOME: state };
  return { root, palette, env };
};

test('local preview reads the real compiler on each HTML request without baking desktop values into builds', async (context) => {
  const { root, palette, env } = await fixture(context);
  const plugin = desktopThemePlugin({ enabled: true, env });
  const server = await createServer({
    root,
    configFile: false,
    logLevel: 'silent',
    plugins: [plugin],
    server: { host: '127.0.0.1', port: 0 },
  });
  context.after(() => server.close());
  await server.listen();
  const origin = server.resolvedUrls.local[0];
  await writeFile(palette, 'mode="dark"\nbackground="#2c2525"\nforeground="#e6d9db"\naccent="#f38d70"\n');
  const first = await (await fetch(origin)).text();
  assert.ok(first.includes('--invokeai-desktop-background:#2c2525'));
  assert.ok(first.includes('--invokeai-desktop-accent:#f38d70'));
  assert.ok(first.includes('<meta name="darkreader-lock">'));
  assert.ok(first.includes('--invokeai-desktop-surface:#373030'));
  assert.ok(first.includes('--invokeai-desktop-selection:#3f3737'));
  assert.equal((first.match(/--invokeai-desktop-/g) ?? []).length, 8);
  await writeFile(palette, 'mode="light"\nbackground="#ffffff"\nforeground="#222222"\naccent="#006b5f"\n');
  const changed = await (await fetch(origin)).text();
  assert.ok(changed.includes('--invokeai-desktop-accent:#006b5f'));
  assert.ok(changed.includes('--invokeai-desktop-color-scheme:light'));
  await writeFile(palette, 'not valid toml!');
  const invalid = await (await fetch(origin)).text();
  assert.ok(invalid.includes(MARKER));
  assert.ok(!invalid.includes('darkreader-lock'));
  await writeFile(palette, 'mode="dark"\nbackground="#2c2525"\nforeground="#e6d9db"\naccent="#f38d70"\n');
  await build({ root, configFile: false, logLevel: 'silent', plugins: [plugin] });
  const built = await readFile(resolve(root, 'dist/index.html'), 'utf8');
  assert.ok(built.includes(MARKER));
  assert.ok(!built.includes('#f38d70'));
  assert.ok(!built.includes('darkreader-lock'));
});

test('desktop preview is opt-in and optional when Python or the palette is unavailable', async (context) => {
  const { root, env } = await fixture(context);
  assert.equal(await desktopThemePlugin().transformIndexHtml(document), document);
  assert.equal(await desktopThemePlugin({ enabled: true, env }).transformIndexHtml(document), document);
  const unavailable = desktopThemePlugin({ enabled: true, env: { ...env, PATH: resolve(root, 'no-python') } });
  assert.equal(await unavailable.transformIndexHtml(document), document);
  assert.equal(await unavailable.transformIndexHtml('<html>no marker</html>'), '<html>no marker</html>');
});
