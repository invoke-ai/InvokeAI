import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

const eslintConfig = readFileSync(new URL('../../eslint.config.mjs', import.meta.url), 'utf8');

describe('eslint config', () => {
  it('includes React Compiler diagnostics from eslint-plugin-react-hooks', () => {
    expect(eslintConfig).toContain("pluginReactHooks.configs['recommended-latest'].rules");
  });
});
