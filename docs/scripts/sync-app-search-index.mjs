// Copies the built docs index (`dist/app-search.json`, from `src/pages/app-search.json.ts`) into the app, which
// commits it for its command palette. `--check` fails instead of writing when the committed copy is out of date.
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

import { formatAppSearchIndex } from '../src/lib/app-search.mjs';

const REGENERATE = 'cd docs && pnpm build && pnpm run sync-app-search-index';
const builtUrl = new URL('../dist/app-search.json', import.meta.url);
const committedPath = 'invokeai/frontend/webv2/src/workbench/palette/docsSearchIndex.json';
const committedUrl = new URL(`../../${committedPath}`, import.meta.url);

if (!existsSync(builtUrl)) {
  console.error(`dist/app-search.json is missing. Build the docs first: ${REGENERATE}`);
  process.exit(1);
}

const expected = formatAppSearchIndex(JSON.parse(readFileSync(builtUrl, 'utf8')));

if (process.argv.includes('--check')) {
  const committed = existsSync(committedUrl) ? readFileSync(committedUrl, 'utf8') : '';
  if (committed !== expected) {
    console.error(`${committedPath} does not match the docs build. Regenerate and commit it: ${REGENERATE}`);
    process.exit(1);
  }
  console.log('The app search index matches the docs build.');
} else {
  writeFileSync(committedUrl, expected);
  console.log(`Wrote ${committedPath}.`);
}
