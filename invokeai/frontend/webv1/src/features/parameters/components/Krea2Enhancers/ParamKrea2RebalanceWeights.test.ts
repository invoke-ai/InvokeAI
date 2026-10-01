import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

const source = readFileSync(new URL('./ParamKrea2RebalanceWeights.tsx', import.meta.url), 'utf8');
const english = JSON.parse(readFileSync(new URL('../../../../../public/locales/en.json', import.meta.url), 'utf8')) as {
  parameters?: Record<string, string>;
};

describe('ParamKrea2RebalanceWeights localisation', () => {
  it('uses a translated placeholder with an English locale entry', () => {
    expect(source).toContain("placeholder={t('parameters.krea2RebalanceWeightsPlaceholder')}");
    expect(english.parameters?.krea2RebalanceWeightsPlaceholder).toBe(
      '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0'
    );
  });
});
