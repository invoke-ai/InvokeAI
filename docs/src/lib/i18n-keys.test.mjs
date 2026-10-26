import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { describe, it } from 'node:test';

import { i18nSchema } from '@astrojs/starlight/schema';

import { translatedLocales } from '../config/locales.mjs';
import { modelFamilies } from '../config/model-families.ts';
import { docsTopics, topicGroups } from '../config/topics.ts';
import en from '../content/i18n/en.json' with { type: 'json' };
import { fileDirectory, sidebarGroupKey } from './sidebar.mjs';
import { stripSortPrefix } from './sort-prefix.mjs';

// `src/content/i18n/en.json` is what Crowdin translates. These tests keep it in step with the code.

const builtInKeys = new Set(Object.keys(i18nSchema().shape));
const { default: builtInEnglish } = await import(
  new URL('./translations/en.js', import.meta.resolve('@astrojs/starlight/schema')).href
);
const srcDirectory = new URL('../', import.meta.url);
const pluralSuffix = /_(zero|one|two|few|many|other)$/;

/** Every source file's text, tests and content aside. */
const sources = readdirSync(srcDirectory, { recursive: true, withFileTypes: true })
  .filter((entry) => entry.isFile() && /\.(astro|ts|mjs)$/.test(entry.name) && !entry.name.endsWith('.test.mjs'))
  .filter((entry) => !entry.parentPath.includes('/src/content'))
  .map((entry) => readFileSync(`${entry.parentPath}/${entry.name}`, 'utf8'));

/** The keys code names literally: `t('key')`, `pluralForms(t, 'key')` and setup options' `labelKey: 'key'`. */
const usedKeys = new Set(
  sources.flatMap((source) =>
    [...source.matchAll(/(?:\bt\(\s*|\bpluralForms\(\s*t,\s*|\blabelKey:\s*)(['"])([^'"]+)\1/g)].map(
      (match) => match[2]
    )
  )
);

/** The English labels of the sidebar groups made from docs directories: every directory inside a topic's own. */
const sidebarGroupLabels = () => {
  const topicDirectories = docsTopics.map((topic) => topic.directory);
  return readdirSync(new URL('../content/docs/', import.meta.url), { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => ({ name: entry.name, directory: fileDirectory(`${entry.parentPath}/${entry.name}/page.md`) }))
    .filter(({ directory }) => !translatedLocales.includes(directory.split('/')[0]))
    .filter(({ directory }) => !directory.split('/').includes('assets'))
    .filter(
      ({ directory }) =>
        !topicDirectories.includes(directory) && topicDirectories.some((topic) => directory.startsWith(`${topic}/`))
    )
    .map(({ name }) => stripSortPrefix(name));
};

describe('en.json', () => {
  it('has every key the code names', () => {
    const missing = [...usedKeys].filter(
      (key) => !(key in en) && !builtInKeys.has(key) && !(`${key}_other` in en)
    );
    assert.deepEqual(missing, []);
  });

  it('has the label, description and group of every topic, and no others', () => {
    const expected = [
      ...docsTopics.flatMap(({ id }) => [`topics.${id}.label`, `topics.${id}.description`]),
      ...topicGroups.map((group) => `topics.group.${group.toLowerCase()}`),
    ];
    assert.deepEqual(Object.keys(en).filter((key) => key.startsWith('topics.')).sort(), expected.sort());
  });

  // starlight-sidebar-topics identifies topics by their English label in `topics.ts`; pages show the UI string.
  it('labels each topic as its sidebar-topics label does', () => {
    const mismatched = docsTopics.filter(({ id, label }) => en[`topics.${id}.label`] !== label).map(({ id }) => id);
    assert.deepEqual(mismatched, []);
  });

  it('has the label of every sidebar group made from a docs directory, and no others', () => {
    const expected = sidebarGroupLabels().map(sidebarGroupKey);
    assert.ok(expected.includes('sidebar.group.image-generation'), 'nested Users Guide sections are groups');
    const keys = Object.keys(en).filter((key) => key.startsWith('sidebar.group.'));
    assert.deepEqual(keys.sort(), [...new Set(expected)].sort());
  });

  it('has the tagline of every model family on the home page, and no others', () => {
    const keys = Object.keys(en).filter((key) => /^modelFamilies\.[^.]+\.tagline$/.test(key));
    assert.deepEqual(keys.sort(), modelFamilies.map(({ slug }) => `modelFamilies.${slug}.tagline`).sort());
  });

  it('has no key the code does not use', () => {
    // Keys the code builds from ids, which the tests above match to those ids.
    const builtByCode = /^(topics\.|sidebar\.group\.|modelFamilies\.[^.]+\.tagline$)/;
    const unused = Object.keys(en)
      .map((key) => key.replace(pluralSuffix, ''))
      .filter((key) => !usedKeys.has(key) && !builtByCode.test(key));
    assert.deepEqual(unused, []);
  });

  // A built-in string in en.json comes back as English in every locale Crowdin has not translated it for, in place of
  // Starlight's own translation.
  it('overrides a built-in Starlight string only to reword it', () => {
    const copies = Object.keys(en).filter((key) => builtInKeys.has(key) && en[key] === builtInEnglish[key]);
    const notInStarlightEnglish = Object.keys(en).filter((key) => builtInKeys.has(key) && !(key in builtInEnglish));
    assert.deepEqual({ copies, notInStarlightEnglish }, { copies: [], notInStarlightEnglish: [] });
  });
});
