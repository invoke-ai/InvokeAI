#!/usr/bin/env node
/* eslint-disable no-console */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class TranslationCleaner {
  constructor(srcDir) {
    this.srcDir = srcDir;
    this.fileCache = new Map();
    this.duplicateKeys = [];
  }

  getKeys(obj, currentPath = '', keys = []) {
    for (const key in obj) {
      const newPath = currentPath ? `${currentPath}.${key}` : key;
      const value = obj[key];

      if (typeof value === 'object' && value !== null) {
        this.getKeys(value, newPath, keys);
      } else {
        // Include all keys - we'll handle pluralization in searchCodebase
        keys.push(newPath);
      }
    }
    return keys;
  }

  searchCodebase(key) {
    // Known i18next pluralization suffixes
    const pluralizationSuffixes = ['_one', '_other', '_zero', '_few', '_many', '_two'];

    // Check if this is a pluralized key
    const lastPart = key.split('.').pop();
    const isPluralizedKey = pluralizationSuffixes.some((suffix) => lastPart.endsWith(suffix));

    // If it's a pluralized key, also check for the base key (without suffix)
    let keysToCheck = [key];
    if (isPluralizedKey) {
      // Extract the base key by removing the pluralization suffix
      const suffixMatch = pluralizationSuffixes.find((suffix) => lastPart.endsWith(suffix));
      if (suffixMatch) {
        const basePart = lastPart.slice(0, -suffixMatch.length);
        const keyParts = key.split('.');
        keyParts[keyParts.length - 1] = basePart;
        const baseKey = keyParts.join('.');
        keysToCheck.push(baseKey);
      }
    }

    const searchDir = (dir) => {
      const files = fs.readdirSync(dir);

      for (const file of files) {
        const fullPath = path.join(dir, file);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
          if (searchDir(fullPath)) {
            return true;
          }
        } else if (file.endsWith('.ts') || file.endsWith('.tsx')) {
          let content;
          if (this.fileCache.has(fullPath)) {
            content = this.fileCache.get(fullPath);
          } else {
            content = fs.readFileSync(fullPath, 'utf8');
            this.fileCache.set(fullPath, content);
          }

          // Check each variant of the key
          for (const keyToCheck of keysToCheck) {
            // Escape special regex characters in the key
            const escapedKey = keyToCheck.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

            // Check for the whole key surrounded by quotes
            if (new RegExp(`['"\`]${escapedKey}['"\`]`).test(content)) {
              return true;
            }

            // Check for the last part of the key (stem) with quotes
            const stem = keyToCheck.split('.').pop();
            const escapedStem = stem.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            if (new RegExp(`${escapedStem}['"\`]`).test(content)) {
              return true;
            }
          }
        }
      }
      return false;
    };

    return searchDir(this.srcDir);
  }

  removeKey(obj, keyPath) {
    const path = keyPath.split('.');
    const lastKey = path.pop();

    let current = obj;
    for (const key of path) {
      current = current[key];
    }

    delete current[lastKey];
  }

  removeEmptyObjects(obj) {
    for (const key in obj) {
      const value = obj[key];
      if (typeof value === 'object' && value !== null) {
        this.removeEmptyObjects(value);
        if (Object.keys(value).length === 0) {
          delete obj[key];
        }
      }
    }
    return obj;
  }

  detectDuplicates(obj, path = '') {
    const seenKeys = new Set();
    const duplicates = [];

    for (const key in obj) {
      const fullPath = path ? `${path}.${key}` : key;

      if (seenKeys.has(key)) {
        duplicates.push(fullPath);
      } else {
        seenKeys.add(key);
      }

      const value = obj[key];
      if (typeof value === 'object' && value !== null) {
        const subDuplicates = this.detectDuplicates(value, fullPath);
        duplicates.push(...subDuplicates);
      }
    }

    return duplicates;
  }

  clean(translations) {
    const keys = this.getKeys(translations);
    const removedKeys = [];

    console.log(`Checking ${keys.length} translation keys...`);

    for (const key of keys) {
      if (!this.searchCodebase(key)) {
        this.removeKey(translations, key);
        removedKeys.push(key);
      }
    }

    if (removedKeys.length > 0) {
      console.log(`\nFound ${removedKeys.length} unused keys:`);
      removedKeys.forEach((key) => console.log(`  - ${key}`));
    } else {
      console.log('No unused keys found');
    }

    // Remove empty objects left after key removal
    this.removeEmptyObjects(translations);

    return { translations, removedCount: removedKeys.length, removedKeys };
  }

  check(translations) {
    const copyTranslations = JSON.parse(JSON.stringify(translations));
    const { removedCount, removedKeys } = this.clean(copyTranslations);
    return { isClean: removedCount === 0, removedKeys };
  }
}

function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    input: null,
    srcDir: null,
    mode: 'check',
    output: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === '--input' && i + 1 < args.length) {
      options.input = args[++i];
    } else if (arg === '--src-dir' && i + 1 < args.length) {
      options.srcDir = args[++i];
    } else if (arg === '--mode' && i + 1 < args.length) {
      options.mode = args[++i];
    } else if (arg === '--output' && i + 1 < args.length) {
      options.output = args[++i];
    } else if (!arg.startsWith('--') && !options.input) {
      options.input = arg;
    }
  }

  if (!options.input) {
    console.error('Error: Input file path is required');
    console.error(
      'Usage: node clean-translations.js <input-file> [--mode check|clean] [--src-dir <dir>] [--output <file>]'
    );
    process.exit(1);
  }

  // Set defaults
  if (!options.srcDir) {
    // Default to ../src relative to the script location
    options.srcDir = path.join(__dirname, '..', 'src');
  }

  if (!options.output) {
    options.output = options.input;
  }

  // Resolve paths to absolute
  options.input = path.resolve(options.input);
  options.srcDir = path.resolve(options.srcDir);
  options.output = path.resolve(options.output);

  return options;
}

function main() {
  const options = parseArgs();

  console.log(`Mode: ${options.mode}`);
  console.log(`Input: ${options.input}`);
  console.log(`Source directory: ${options.srcDir}`);

  if (!fs.existsSync(options.input)) {
    console.error(`Error: Input file not found: ${options.input}`);
    process.exit(1);
  }

  if (!fs.existsSync(options.srcDir)) {
    console.error(`Error: Source directory not found: ${options.srcDir}`);
    process.exit(1);
  }

  const cleaner = new TranslationCleaner(options.srcDir);
  let jsonString = fs.readFileSync(options.input, 'utf8');

  const translations = JSON.parse(jsonString);

  if (options.mode === 'check') {
    console.log('Checking for unused translations...');
    const { isClean } = cleaner.check(translations);

    if (isClean) {
      console.log('✓ All translations are in use');
      process.exit(0);
    } else {
      console.error('\n✗ Found unused translations. Run with --mode clean to remove them.');
      process.exit(1);
    }
  } else if (options.mode === 'clean') {
    console.log('Cleaning unused translations...');
    const { translations: cleanedTranslations } = cleaner.clean(translations);

    console.log(`\nWriting cleaned translations to: ${options.output}`);
    fs.writeFileSync(options.output, `${JSON.stringify(cleanedTranslations, null, 2)}\n`);
    console.log('✓ Translations cleaned successfully');
    console.log('\nNext step: Run `pnpm fix` to ensure proper formatting');
    process.exit(0);
  } else {
    console.error(`Error: Invalid mode '${options.mode}'. Use 'check' or 'clean'.`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
