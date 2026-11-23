import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { it } from 'node:test';

// The home page shows the app's V7 logo. Vite's dev server refuses files outside the docs package, so the docs keep
// a copy; this keeps the copy identical to the app's file.
it('keeps the V7 logo identical to the app', () => {
  const docsCopy = readFileSync(new URL('../assets/v7-logo.webp', import.meta.url));
  const appOriginal = readFileSync(
    new URL('../../../invokeai/frontend/webv2/src/assets/V7Logo.webp', import.meta.url),
  );
  assert.ok(docsCopy.equals(appOriginal), 'Copy invokeai/frontend/webv2/src/assets/V7Logo.webp to docs/src/assets/v7-logo.webp.');
});
