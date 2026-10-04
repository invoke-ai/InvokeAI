/** Supported application languages and their document direction. */
export const WORKBENCH_LANGUAGE_OPTIONS = [
  { value: 'ar', label: 'العربية', direction: 'rtl' },
  { value: 'az', label: 'Azərbaycan dili', direction: 'ltr' },
  { value: 'bg', label: 'Български', direction: 'ltr' },
  { value: 'de', label: 'Deutsch', direction: 'ltr' },
  { value: 'en', label: 'English', direction: 'ltr' },
  { value: 'en-GB', label: 'English (UK)', direction: 'ltr' },
  { value: 'es', label: 'Español', direction: 'ltr' },
  { value: 'fi', label: 'Suomi', direction: 'ltr' },
  { value: 'fr', label: 'Français', direction: 'ltr' },
  { value: 'he', label: 'עִבְֿרִית', direction: 'rtl' },
  { value: 'hu', label: 'Magyar Nyelv', direction: 'ltr' },
  { value: 'it', label: 'Italiano', direction: 'ltr' },
  { value: 'ja', label: '日本語', direction: 'ltr' },
  { value: 'ko', label: '한국어', direction: 'ltr' },
  { value: 'mn', label: 'Монгол', direction: 'ltr' },
  { value: 'nl', label: 'Nederlands', direction: 'ltr' },
  { value: 'pl', label: 'Polski', direction: 'ltr' },
  { value: 'pt', label: 'Português', direction: 'ltr' },
  { value: 'pt-BR', label: 'Português do Brasil', direction: 'ltr' },
  { value: 'ro', label: 'Română', direction: 'ltr' },
  { value: 'ru', label: 'Русский', direction: 'ltr' },
  { value: 'sv', label: 'Svenska', direction: 'ltr' },
  { value: 'tr', label: 'Türkçe', direction: 'ltr' },
  { value: 'uk', label: 'Українська', direction: 'ltr' },
  { value: 'vi', label: 'Tiếng Việt', direction: 'ltr' },
  { value: 'zh-CN', label: '简体中文', direction: 'ltr' },
  { value: 'zh-Hant', label: '漢語', direction: 'ltr' },
] as const;

export type WorkbenchLanguage = (typeof WORKBENCH_LANGUAGE_OPTIONS)[number]['value'];
export type WorkbenchLanguageDirection = (typeof WORKBENCH_LANGUAGE_OPTIONS)[number]['direction'];

export const WORKBENCH_LANGUAGES: WorkbenchLanguage[] = WORKBENCH_LANGUAGE_OPTIONS.map((option) => option.value);

export const normalizeWorkbenchLanguage = (value: unknown): WorkbenchLanguage | null => {
  if (value === 'ua') {
    return 'uk';
  }

  return typeof value === 'string' && WORKBENCH_LANGUAGES.includes(value as WorkbenchLanguage)
    ? (value as WorkbenchLanguage)
    : null;
};

export const getWorkbenchLanguageDirection = (language: WorkbenchLanguage): WorkbenchLanguageDirection =>
  WORKBENCH_LANGUAGE_OPTIONS.find((option) => option.value === language)?.direction ?? 'ltr';

const BYTE_UNITS = ['byte', 'kilobyte', 'megabyte', 'gigabyte', 'terabyte', 'petabyte'] as const;
const BINARY_BYTE_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'] as const;

const documentLocale = (): string | undefined => globalThis.document?.documentElement.lang || undefined;

// Constructing a NumberFormat costs far more than formatting with one, and these run per row in lists.
const numberFormats = new Map<string, Intl.NumberFormat>();
const getNumberFormat = (locale: string | undefined, options: Intl.NumberFormatOptions): Intl.NumberFormat => {
  const key = `${locale ?? ''}\u0000${JSON.stringify(options)}`;
  let format = numberFormats.get(key);
  if (!format) {
    format = new Intl.NumberFormat(locale, options);
    numberFormats.set(key, format);
  }
  return format;
};

/** A whole count in the document language's digits and grouping. */
export const formatCount = (count: number, locale: string | undefined = documentLocale()): string =>
  getNumberFormat(locale, {}).format(count);

/**
 * A byte size with one decimal from kilobytes up, in the document language by default. Decimal (SI) units are the
 * ones Intl can name; `binary` scales by 1024 and appends IEC labels (KiB, GiB) that Intl lacks. Missing, negative or
 * non-finite sizes render as a dash.
 */
export const formatBytes = (
  bytes: number | null | undefined,
  { binary = false, locale = documentLocale() }: { binary?: boolean; locale?: string } = {}
): string => {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes < 0) {
    return '—';
  }

  const base = binary ? 1024 : 1000;
  let value = bytes;
  let unit = 0;

  while (Math.round(value * 10) / 10 >= base && unit < BYTE_UNITS.length - 1) {
    value /= base;
    unit += 1;
  }

  const digits = unit === 0 ? 0 : 1;
  if (binary) {
    const number = getNumberFormat(locale, { maximumFractionDigits: digits, minimumFractionDigits: digits });
    return `${number.format(value)}\u00a0${BINARY_BYTE_UNITS[unit]}`;
  }
  return getNumberFormat(locale, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
    style: 'unit',
    unit: BYTE_UNITS[unit],
    unitDisplay: unit === 0 ? 'long' : 'short',
  }).format(value);
};
