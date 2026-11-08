import assert from 'node:assert/strict';
import { globSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

// The Crowdin CLI fails on a `crowdin.yml` source pattern that matches no file, and the `.md` entry rests on a single
// page.

const repoRoot = fileURLToPath(new URL('../../../', import.meta.url));
const config = readFileSync(path.join(repoRoot, 'crowdin.yml'), 'utf8');
const sources = [...config.matchAll(/^\s*-?\s*source:\s*'([^']+)'/gm)].map((match) => match[1]);

describe('crowdin.yml', () => {
  it('has a source pattern for every entry', () => {
    assert.equal(sources.length, [...config.matchAll(/^\s*-?\s*source:/gm)].length);
  });

  for (const source of sources) {
    it(`matches pages with ${source}`, () => {
      const matches = globSync(source.replace(/^\//, ''), { cwd: repoRoot }).filter((file) => !file.includes('/assets/'));
      assert.ok(matches.length > 0, `${source} matches no file`);
    });
  }
});
