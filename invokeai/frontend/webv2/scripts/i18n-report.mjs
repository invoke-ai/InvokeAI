/* eslint-disable no-console */

import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const localeDir = join(process.cwd(), 'public/locales');
const verbose = process.argv.includes('--verbose');

const flattenKeys = (value, prefix = '') => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return prefix ? [prefix] : [];
  }

  return Object.entries(value).flatMap(([key, child]) => flattenKeys(child, prefix ? `${prefix}.${key}` : key));
};

const readJson = async (fileName) => JSON.parse(await readFile(join(localeDir, fileName), 'utf8'));

const localeFiles = (await readdir(localeDir)).filter((fileName) => fileName.endsWith('.json')).sort();
const english = await readJson('en.json');
const englishKeys = flattenKeys(english);
const englishKeySet = new Set(englishKeys);
let missingCount = 0;

for (const fileName of localeFiles) {
  if (fileName === 'en.json') {
    continue;
  }

  const locale = await readJson(fileName);
  const localeKeys = new Set(flattenKeys(locale));
  const missing = englishKeys.filter((key) => !localeKeys.has(key));
  const extra = [...localeKeys].filter((key) => !englishKeySet.has(key));

  missingCount += missing.length;
  console.log(`\n${fileName}`);
  console.log(`  missing: ${missing.length}`);
  console.log(`  extra: ${extra.length}`);

  if (verbose) {
    for (const key of missing) {
      console.log(`    - ${key}`);
    }

    for (const key of extra) {
      console.log(`    + ${key}`);
    }
  }
}

console.log(`\nTotal missing keys: ${missingCount}`);
