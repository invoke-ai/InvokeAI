import { describe, expect, it } from 'vitest';

import { parseCategoryFromName } from './modelPickerCategory';

describe('parseCategoryFromName', () => {
  it('returns no category when there is no bracket prefix', () => {
    expect(parseCategoryFromName('flux.1 dev')).toEqual({ category: null, displayName: 'flux.1 dev' });
  });

  it('extracts the category and strips the prefix from the display name', () => {
    expect(parseCategoryFromName('[flux]flux.1 dev')).toEqual({ category: 'flux', displayName: 'flux.1 dev' });
  });

  it('tolerates whitespace inside and around the bracket prefix', () => {
    expect(parseCategoryFromName('  [ flux ]  flux.1 dev ')).toEqual({ category: 'flux', displayName: 'flux.1 dev' });
  });

  it('treats an empty bracket as no category', () => {
    expect(parseCategoryFromName('[]flux.1 dev')).toEqual({ category: null, displayName: '[]flux.1 dev' });
  });

  it('falls back to the original name when the prefix is present but the rest is empty', () => {
    expect(parseCategoryFromName('[flux]')).toEqual({ category: 'flux', displayName: '[flux]' });
  });

  it('only matches a bracket prefix at the start of the name', () => {
    expect(parseCategoryFromName('flux [v2]')).toEqual({ category: null, displayName: 'flux [v2]' });
  });
});
