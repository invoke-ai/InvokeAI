import type { S } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, expect, it } from 'vitest';

import {
  getImageSubfolderStrategyOption,
  imageSubfolderStrategyOptions,
  isImageSubfolderStrategy,
} from './SettingsImageSubfolderStrategySelect';

describe('image subfolder strategy settings options', () => {
  it('covers all runtime config image subfolder strategies exposed by the API', () => {
    type ImageSubfolderStrategy = NonNullable<S['UpdateAppGenerationSettingsRequest']['image_subfolder_strategy']>;
    type OptionValue = (typeof imageSubfolderStrategyOptions)[number]['value'];

    assert<Equals<Exclude<ImageSubfolderStrategy, OptionValue>, never>>();
    assert<Equals<Exclude<OptionValue, ImageSubfolderStrategy>, never>>();
  });

  it('includes all runtime config image subfolder strategies', () => {
    expect(imageSubfolderStrategyOptions.map((option) => option.value)).toEqual(['flat', 'date', 'type', 'hash']);
  });

  it('validates image subfolder strategy values', () => {
    expect(isImageSubfolderStrategy('date')).toBe(true);
    expect(isImageSubfolderStrategy('unknown')).toBe(false);
  });

  it('gets the option for the active strategy', () => {
    expect(getImageSubfolderStrategyOption('hash')).toEqual({
      label: 'settings.imageSubfolderStrategyHash',
      value: 'hash',
    });
  });

  it('gets an explicit unknown option for an unrecognized active strategy', () => {
    expect(getImageSubfolderStrategyOption('unknown')).toEqual({
      label: 'settings.imageSubfolderStrategyUnknown',
      value: 'unknown',
    });
  });
});
