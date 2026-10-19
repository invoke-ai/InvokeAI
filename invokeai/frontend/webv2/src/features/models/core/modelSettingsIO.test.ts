import { describe, expect, it } from 'vitest';

import type { ModelConfig } from './types';

import {
  buildExportData,
  dataUrlToFile,
  isImageDataUrl,
  isSafeUrl,
  partitionImportableFields,
  sanitizeFilename,
  validateImportData,
} from './modelSettingsIO';

// 1x1 transparent PNG.
const PNG_DATA_URL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

const model = (overrides: Partial<ModelConfig>): ModelConfig =>
  ({
    base: 'sdxl',
    file_size: 1,
    format: 'checkpoint',
    hash: 'hash',
    key: 'key',
    name: 'Model',
    path: '/models/m.safetensors',
    source: '/models/m.safetensors',
    source_type: 'path',
    type: 'main',
    ...overrides,
  }) as ModelConfig;

describe('sanitizeFilename', () => {
  it('replaces filesystem-unsafe characters with underscores', () => {
    expect(sanitizeFilename('a/b\\c:d*e?f"g<h>i|j')).toBe('a_b_c_d_e_f_g_h_i_j');
    expect(sanitizeFilename('Juggernaut XL v9')).toBe('Juggernaut XL v9');
  });
});

describe('isSafeUrl', () => {
  it('accepts only http(s)', () => {
    expect(isSafeUrl('https://example.com')).toBe(true);
    expect(isSafeUrl('http://example.com')).toBe(true);
    expect(isSafeUrl('javascript:alert(1)')).toBe(false);
    expect(isSafeUrl('file:///etc/passwd')).toBe(false);
  });
});

describe('isImageDataUrl / dataUrlToFile', () => {
  it('accepts base64 image data URLs and rejects everything else', () => {
    expect(isImageDataUrl(PNG_DATA_URL)).toBe(true);
    expect(isImageDataUrl('data:text/html;base64,PGI+')).toBe(false);
    expect(isImageDataUrl('data:image/png,not-base64')).toBe(false);
    expect(isImageDataUrl('https://example.com/x.png')).toBe(false);
  });

  it('decodes a valid data URL into a File and returns null otherwise', () => {
    const file = dataUrlToFile(PNG_DATA_URL, 'cover.png');

    expect(file).toBeInstanceOf(File);
    expect(file?.type).toBe('image/png');
    expect(file?.name).toBe('cover.png');

    expect(dataUrlToFile('not-a-data-url', 'x.png')).toBeNull();
    expect(dataUrlToFile('data:image/png;base64,', 'x.png')).toBeNull();
    expect(dataUrlToFile('data:image/png;base64,!!!not-base64!!!', 'x.png')).toBeNull();
  });
});

describe('buildExportData', () => {
  it('includes only present, non-empty fields', () => {
    expect(buildExportData(model({ description: null, name: '' }))).toEqual({});

    expect(
      buildExportData(
        model({
          cpu_only: false,
          default_settings: { weight: 0.8 },
          description: 'desc',
          name: 'Named',
          source_url: 'https://example.com',
          trigger_phrases: ['t1'],
        })
      )
    ).toEqual({
      cpu_only: false,
      default_settings: { weight: 0.8 },
      description: 'desc',
      name: 'Named',
      source_url: 'https://example.com',
      trigger_phrases: ['t1'],
    });
  });

  it('never includes the cover image (fetched separately, async)', () => {
    expect(buildExportData(model({ cover_image: 'present', name: 'X' }))).toEqual({ name: 'X' });
  });
});

describe('validateImportData', () => {
  it('accepts an empty object and a fully populated export, rejects non-objects', () => {
    expect(validateImportData({})).toBe(true);
    expect(
      validateImportData({
        cover_image: PNG_DATA_URL,
        cpu_only: true,
        default_settings: { weight: 1 },
        description: 'd',
        name: 'n',
        source_url: 'https://example.com',
        trigger_phrases: ['a', 'b'],
        unknown_field: 'ignored',
      })
    ).toBe(true);
    expect(validateImportData(null)).toBe(false);
    expect(validateImportData([])).toBe(false);
    expect(validateImportData('json')).toBe(false);
  });

  it('rejects wrong field types and unsafe values', () => {
    expect(validateImportData({ name: 42 })).toBe(false);
    expect(validateImportData({ description: {} })).toBe(false);
    expect(validateImportData({ source_url: 'javascript:alert(1)' })).toBe(false);
    expect(validateImportData({ source_url: '' })).toBe(true);
    expect(validateImportData({ cover_image: 'https://example.com/x.png' })).toBe(false);
    expect(validateImportData({ trigger_phrases: ['ok', 5] })).toBe(false);
    expect(validateImportData({ default_settings: [] })).toBe(false);
    expect(validateImportData({ cpu_only: 'yes' })).toBe(false);
  });
});

describe('partitionImportableFields', () => {
  it('applies fields the target config carries and skips the rest', () => {
    const target = model({ default_settings: null, description: null, trigger_phrases: null });
    // cpu_only is absent from this main-model config entirely.
    const { body, skipped } = partitionImportableFields(
      { cpu_only: true, default_settings: { steps: 20 }, name: 'Renamed', trigger_phrases: ['x'] },
      target
    );

    expect(body).toEqual({ default_settings: { steps: 20 }, name: 'Renamed', trigger_phrases: ['x'] });
    expect(skipped).toEqual(['cpu_only']);
  });

  it('ignores null and undefined import values', () => {
    const { body, skipped } = partitionImportableFields({ description: null, name: undefined }, model({}));

    expect(body).toEqual({});
    expect(skipped).toEqual([]);
  });
});
