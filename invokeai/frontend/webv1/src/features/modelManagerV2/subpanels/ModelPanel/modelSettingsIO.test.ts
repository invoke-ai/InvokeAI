import type { AnyModelConfigWithExternal } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import {
  buildExportData,
  dataUrlToFile,
  isImageDataUrl,
  isSafeUrl,
  sanitizeFilename,
  validateImportData,
} from './modelSettingsIO';

const makeConfig = (overrides: Record<string, unknown> = {}): AnyModelConfigWithExternal => {
  return {
    key: 'test-key',
    hash: 'abc123',
    path: '/models/test.safetensors',
    file_size: 0,
    name: 'Test Model',
    description: null,
    source: '/models/test.safetensors',
    source_type: 'path',
    source_api_response: null,
    source_url: null,
    cover_image: null,
    base: 'sd-1',
    type: 'main',
    format: 'checkpoint',
    default_settings: null,
    trigger_phrases: null,
    cpu_only: null,
    ...overrides,
  } as unknown as AnyModelConfigWithExternal;
};

describe('sanitizeFilename', () => {
  it('replaces filesystem-unsafe characters with underscores', () => {
    expect(sanitizeFilename('foo<bar>baz')).toBe('foo_bar_baz');
    expect(sanitizeFilename('a/b\\c:d|e?f*g"h')).toBe('a_b_c_d_e_f_g_h');
  });

  it('leaves safe filenames untouched', () => {
    expect(sanitizeFilename('My Model v2.1')).toBe('My Model v2.1');
  });
});

describe('isSafeUrl', () => {
  it('accepts http and https URLs', () => {
    expect(isSafeUrl('https://example.com')).toBe(true);
    expect(isSafeUrl('http://example.com')).toBe(true);
  });

  it('rejects other schemes', () => {
    expect(isSafeUrl('javascript:alert(1)')).toBe(false);
    expect(isSafeUrl('data:text/html,foo')).toBe(false);
    expect(isSafeUrl('ftp://example.com')).toBe(false);
    expect(isSafeUrl('example.com')).toBe(false);
    expect(isSafeUrl('')).toBe(false);
  });
});

describe('isImageDataUrl', () => {
  it('accepts data URLs with image MIME types', () => {
    expect(isImageDataUrl('data:image/png;base64,iVBORw0KGgo=')).toBe(true);
    expect(isImageDataUrl('data:image/webp;base64,UklGRg==')).toBe(true);
    expect(isImageDataUrl('data:image/jpeg;base64,/9j/4AAQ')).toBe(true);
    expect(isImageDataUrl('data:image/svg+xml;base64,PHN2Zw==')).toBe(true);
  });

  it('rejects non-image data URLs', () => {
    expect(isImageDataUrl('data:text/plain;base64,aGVsbG8=')).toBe(false);
    expect(isImageDataUrl('data:application/json;base64,e30=')).toBe(false);
  });

  it('rejects non-base64 image data URLs', () => {
    expect(isImageDataUrl('data:image/png,iVBORw0KGgo=')).toBe(false);
  });

  it('rejects malformed inputs', () => {
    expect(isImageDataUrl('https://example.com/img.png')).toBe(false);
    expect(isImageDataUrl('not a url at all')).toBe(false);
    expect(isImageDataUrl('')).toBe(false);
  });
});

describe('dataUrlToFile', () => {
  it('decodes a valid base64 image data URL into a File', () => {
    // 1x1 transparent PNG
    const dataUrl =
      'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
    const file = dataUrlToFile(dataUrl, 'thumb.png');
    expect(file).not.toBeNull();
    expect(file?.name).toBe('thumb.png');
    expect(file?.type).toBe('image/png');
    expect(file?.size).toBeGreaterThan(0);
  });

  it('returns null when the prefix is missing', () => {
    expect(dataUrlToFile('iVBORw0KGgo=', 'thumb.png')).toBeNull();
  });

  it('returns null when the data URL has no base64 payload', () => {
    expect(dataUrlToFile('data:image/png;base64,', 'thumb.png')).toBeNull();
  });

  it('returns null on invalid base64 content', () => {
    expect(dataUrlToFile('data:image/png;base64,!!!not-base64!!!', 'thumb.png')).toBeNull();
  });
});

describe('buildExportData', () => {
  it('returns an empty object when no exportable fields are set', () => {
    const config = makeConfig({ name: '' });
    expect(buildExportData(config)).toEqual({});
  });

  it('includes name, description, and source_url when present', () => {
    const config = makeConfig({
      name: 'My LoRA',
      description: 'A cool LoRA',
      source_url: 'https://civitai.com/models/12345',
    });
    expect(buildExportData(config)).toEqual({
      name: 'My LoRA',
      description: 'A cool LoRA',
      source_url: 'https://civitai.com/models/12345',
    });
  });

  it('omits empty string metadata fields', () => {
    const config = makeConfig({
      name: 'My LoRA',
      description: '',
      source_url: '',
    });
    expect(buildExportData(config)).toEqual({ name: 'My LoRA' });
  });

  it('includes trigger_phrases, default_settings, and cpu_only when present', () => {
    const triggerPhrases = ['foo', 'bar'];
    const defaultSettings = { steps: 30 };
    const config = makeConfig({
      name: '',
      trigger_phrases: triggerPhrases,
      default_settings: defaultSettings,
      cpu_only: true,
    });
    expect(buildExportData(config)).toEqual({
      trigger_phrases: triggerPhrases,
      default_settings: defaultSettings,
      cpu_only: true,
    });
  });

  it('includes cpu_only: false (only null is omitted)', () => {
    const config = makeConfig({
      name: '',
      cpu_only: false,
    });
    expect(buildExportData(config)).toEqual({ cpu_only: false });
  });

  it('does not include cover_image (handled separately, async)', () => {
    const config = makeConfig({
      name: 'X',
      cover_image: 'https://example.com/img.png',
    });
    expect(buildExportData(config)).toEqual({ name: 'X' });
  });
});

describe('validateImportData', () => {
  it('accepts an empty object', () => {
    expect(validateImportData({})).toBe(true);
  });

  it('rejects non-object inputs', () => {
    expect(validateImportData(null)).toBe(false);
    expect(validateImportData(undefined)).toBe(false);
    expect(validateImportData('string')).toBe(false);
    expect(validateImportData(42)).toBe(false);
    expect(validateImportData([])).toBe(false);
  });

  it('accepts valid metadata fields', () => {
    expect(
      validateImportData({
        name: 'My Model',
        description: 'a model',
        source_url: 'https://example.com',
      })
    ).toBe(true);
  });

  it('accepts null metadata fields', () => {
    expect(validateImportData({ name: null, description: null, source_url: null })).toBe(true);
  });

  it('rejects non-string name and description', () => {
    expect(validateImportData({ name: 123 })).toBe(false);
    expect(validateImportData({ description: { not: 'a string' } })).toBe(false);
  });

  it('rejects source_url that is not http(s)', () => {
    expect(validateImportData({ source_url: 'javascript:alert(1)' })).toBe(false);
    expect(validateImportData({ source_url: 'ftp://example.com' })).toBe(false);
    expect(validateImportData({ source_url: 'example.com' })).toBe(false);
  });

  it('accepts an empty source_url string', () => {
    expect(validateImportData({ source_url: '' })).toBe(true);
  });

  it('accepts a valid image data URL for cover_image', () => {
    expect(validateImportData({ cover_image: 'data:image/png;base64,iVBORw0KGgo=' })).toBe(true);
  });

  it('rejects non-image cover_image values', () => {
    expect(validateImportData({ cover_image: 'https://example.com/img.png' })).toBe(false);
    expect(validateImportData({ cover_image: 'data:text/plain;base64,aGk=' })).toBe(false);
    expect(validateImportData({ cover_image: 42 })).toBe(false);
  });

  it('validates trigger_phrases as an array of strings', () => {
    expect(validateImportData({ trigger_phrases: ['a', 'b'] })).toBe(true);
    expect(validateImportData({ trigger_phrases: [] })).toBe(true);
    expect(validateImportData({ trigger_phrases: ['a', 1] })).toBe(false);
    expect(validateImportData({ trigger_phrases: 'not-an-array' })).toBe(false);
  });

  it('validates default_settings as a plain object', () => {
    expect(validateImportData({ default_settings: { steps: 30 } })).toBe(true);
    expect(validateImportData({ default_settings: {} })).toBe(true);
    expect(validateImportData({ default_settings: [] })).toBe(false);
    expect(validateImportData({ default_settings: 'nope' })).toBe(false);
  });

  it('validates cpu_only as a boolean', () => {
    expect(validateImportData({ cpu_only: true })).toBe(true);
    expect(validateImportData({ cpu_only: false })).toBe(true);
    expect(validateImportData({ cpu_only: 'true' })).toBe(false);
    expect(validateImportData({ cpu_only: 1 })).toBe(false);
  });

  it('accepts a fully populated valid export', () => {
    expect(
      validateImportData({
        name: 'My Model',
        description: 'desc',
        source_url: 'https://civitai.com/models/1',
        cover_image: 'data:image/webp;base64,UklGRg==',
        trigger_phrases: ['trigger'],
        default_settings: { steps: 30 },
        cpu_only: false,
      })
    ).toBe(true);
  });

  it('ignores unknown fields', () => {
    expect(validateImportData({ name: 'X', someUnknownField: 'whatever' })).toBe(true);
  });
});
