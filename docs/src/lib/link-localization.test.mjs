import assert from 'node:assert/strict';
import test from 'node:test';

import { LOCALE_EXEMPT_PATHS, localizeRootPath, shouldLocalizeRootPath } from './link-localization.mjs';

test('prefixes root-relative docs paths with the locale', () => {
  assert.equal(localizeRootPath('/concepts/prompting-guide/', 'es'), '/es/concepts/prompting-guide/');
  assert.equal(localizeRootPath('/', 'es'), '/es/');
});

test('leaves paths that are not root-relative alone', () => {
  assert.equal(localizeRootPath('https://example.com/', 'es'), 'https://example.com/');
  assert.equal(localizeRootPath('//cdn.example.com/x.js', 'es'), '//cdn.example.com/x.js');
  assert.equal(localizeRootPath('#section', 'es'), '#section');
  assert.equal(localizeRootPath('./assets/gallery.png', 'es'), './assets/gallery.png');
});

test('does not double-prefix an already localized path', () => {
  assert.equal(localizeRootPath('/es', 'es'), '/es');
  assert.equal(localizeRootPath('/es/features/gallery/', 'es'), '/es/features/gallery/');
  // A different locale's path is still a valid target and must not be re-prefixed twice.
  assert.equal(localizeRootPath('/de/features/gallery/', 'de'), '/de/features/gallery/');
});

test('never localizes custom pages that have no per-locale route', () => {
  for (const exempt of LOCALE_EXEMPT_PATHS) {
    assert.equal(shouldLocalizeRootPath(exempt, 'hi'), false);
    assert.equal(shouldLocalizeRootPath(`${exempt}/`, 'hi'), false);
    assert.equal(localizeRootPath(exempt, 'hi'), exempt);
  }
});

test('does not treat a path that merely starts with an exempt path as exempt', () => {
  // `/downloads/` is a docs route, not the `/download` custom page.
  assert.equal(localizeRootPath('/downloads/', 'hi'), '/hi/downloads/');
  // Same for a locale-like prefix that is not the locale segment.
  assert.equal(localizeRootPath('/development/guides/', 'de'), '/de/development/guides/');
});
