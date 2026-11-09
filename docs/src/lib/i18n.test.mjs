import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { formatPlural, interpolate, parseTags, pluralForms } from './i18n.mjs';

describe('parseTags', () => {
  it('splits a message around its tags, wherever the translation puts them', () => {
    assert.deepEqual(parseTags('Built by <link>Aether Fox Studio</link>.'), [
      { text: 'Built by ' },
      { tag: 'link', text: 'Aether Fox Studio' },
      { text: '.' },
    ]);
    assert.deepEqual(parseTags('<link>Aether Fox Studio</link> ने बनाया'), [
      { tag: 'link', text: 'Aether Fox Studio' },
      { text: ' ने बनाया' },
    ]);
    assert.deepEqual(parseTags('Aktualisiert am <time>1. Okt.</time>'), [
      { text: 'Aktualisiert am ' },
      { tag: 'time', text: '1. Okt.' },
    ]);
  });

  it('keeps an empty tag, which a script fills in later', () => {
    assert.deepEqual(parseTags('Showing commands for <setup></setup>'), [
      { text: 'Showing commands for ' },
      { tag: 'setup', text: '' },
    ]);
  });

  it('leaves unclosed and mismatched tags as literal text', () => {
    for (const message of [
      'Report <link>an issue',
      'Report <link>an issue</lnk>',
      'a < b > c',
    ]) {
      assert.deepEqual(parseTags(message), [{ text: message }]);
    }
  });
});

describe('interpolate', () => {
  it('fills known placeholders and leaves the others', () => {
    assert.equal(
      interpolate('No results for “{{ query }}” in {{topic}}.', { query: '<x>', topic: 'Models' }),
      'No results for “<x>” in Models.'
    );
    assert.equal(interpolate('default {{value}} of {{missing}}', { value: 0 }), 'default 0 of {{missing}}');
  });
});

describe('plurals', () => {
  const english = { results_one: '{{count}} result', results_other: '{{count}} results' };
  /** A `t` over a locale's strings with the English fallback, as Starlight's i18next instance resolves them. */
  const translate = (strings) => {
    const lookup = (key) => strings[key] ?? english[key];
    return Object.assign(lookup, { exists: (key) => lookup(key) !== undefined });
  };

  it("collects the locale's forms, including categories English lacks", () => {
    const spanish = {
      results_one: '{{count}} resultado',
      results_many: '{{count}} de resultados',
      results_other: '{{count}} resultados',
    };
    assert.deepEqual(pluralForms(translate(spanish), 'results'), {
      one: '{{count}} resultado',
      many: '{{count}} de resultados',
      other: '{{count}} resultados',
    });
    assert.deepEqual(pluralForms(translate({}), 'results'), { one: '{{count}} result', other: '{{count}} results' });
  });

  it("picks the form the language's plural rules call for", () => {
    const forms = { one: '{{count}} result', other: '{{count}} results' };
    assert.equal(formatPlural(forms, 1, 'en'), '1 result');
    assert.equal(formatPlural(forms, 0, 'en'), '0 results');
    // Hindi treats 0 as singular.
    assert.equal(formatPlural({ one: '{{count}} परिणाम', other: '{{count}} परिणामों' }, 0, 'hi'), '0 परिणाम');
    const spanish = { one: '{{count}} resultado', many: '{{count}} de resultados', other: '{{count}} resultados' };
    assert.equal(formatPlural(spanish, 1000000, 'es'), '1000000 de resultados');
    assert.equal(formatPlural(spanish, 3, 'es'), '3 resultados');
  });

  it('falls back to the other form when the translation lacks the chosen category', () => {
    // Russian picks `few` for 2.
    const russian = { one: '{{count}} результат', other: '{{count}} результатов' };
    assert.equal(formatPlural(russian, 2, 'ru'), '2 результатов');
  });
});
