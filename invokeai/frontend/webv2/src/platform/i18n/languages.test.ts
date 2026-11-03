import { describe, expect, it } from 'vitest';

import { formatBytes, formatCount } from './languages';

describe('formatBytes', () => {
  it('renders a dash for missing or invalid sizes', () => {
    expect(formatBytes(null)).toBe('—');
    expect(formatBytes(undefined)).toBe('—');
    expect(formatBytes(-1)).toBe('—');
    expect(formatBytes(Number.NaN)).toBe('—');
  });

  it('scales across decimal unit boundaries', () => {
    expect(formatBytes(0, { locale: 'en' })).toBe('0 bytes');
    expect(formatBytes(999, { locale: 'en' })).toBe('999 bytes');
    expect(formatBytes(1000, { locale: 'en' })).toBe('1.0 kB');
    expect(formatBytes(999_960, { locale: 'en' })).toBe('1.0 MB');
    expect(formatBytes(1_500_000_000, { locale: 'en' })).toBe('1.5 GB');
  });

  it('follows the locale for separators and unit names', () => {
    expect(formatBytes(1_500_000, { locale: 'de' })).toBe('1,5 MB');
    expect(formatBytes(1_500_000, { locale: 'fr' }).replace(/\s/gu, ' ')).toBe('1,5 Mo');
  });

  it('scales binary sizes by 1024 under IEC labels, with localized digits', () => {
    expect(formatBytes(512, { binary: true, locale: 'en' })).toBe('512\u00a0B');
    expect(formatBytes(1024, { binary: true, locale: 'en' })).toBe('1.0\u00a0KiB');
    expect(formatBytes(24 * 2 ** 30, { binary: true, locale: 'en' })).toBe('24.0\u00a0GiB');
    expect(formatBytes(1.5 * 2 ** 20, { binary: true, locale: 'de' })).toBe('1,5\u00a0MiB');
  });
});

describe('formatCount', () => {
  it('groups digits for the locale', () => {
    expect(formatCount(12_345, 'en')).toBe('12,345');
    expect(formatCount(12_345, 'de')).toBe('12.345');
  });
});
