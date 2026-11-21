export const IMAGE_EXPORT_FORMATS = ['png', 'jpeg', 'webp', 'avif'] as const;

export type ImageExportFormat = (typeof IMAGE_EXPORT_FORMATS)[number];

export const IMAGE_EXPORT_METADATA = ['all', 'parameters', 'none'] as const;

export type ImageExportMetadata = (typeof IMAGE_EXPORT_METADATA)[number];

export interface ImageExportSettings {
  format: ImageExportFormat;
  /** 1..100, read by the lossy formats only. */
  quality: number;
  /** A cap on the longer side in px; null keeps the original size. Export never enlarges. */
  longEdge: number | null;
  /** Read by PNG only: the other formats always export without generation data. */
  metadata: ImageExportMetadata;
  namePattern: string;
}

export const DEFAULT_IMAGE_EXPORT_SETTINGS: ImageExportSettings = {
  format: 'png',
  longEdge: null,
  metadata: 'all',
  namePattern: '{name}',
  quality: 90,
};

/** The server's pattern limit (`ImageExportNamePattern`). */
export const IMAGE_EXPORT_NAME_PATTERN_MAX_LENGTH = 200;

export const IMAGE_EXPORT_NAME_TOKENS = ['{name}', '{board}', '{seed}', '{model}', '{date}', '{n}'] as const;

export const IMAGE_EXPORT_FORMAT_LABELS: Readonly<Record<ImageExportFormat, string>> = {
  avif: 'AVIF',
  jpeg: 'JPEG',
  png: 'PNG',
  webp: 'WebP',
};

export const isLossyImageExportFormat = (format: ImageExportFormat): boolean => format !== 'png';

const isOneOf = <Value extends string>(values: readonly Value[], value: unknown): value is Value =>
  typeof value === 'string' && (values as readonly string[]).includes(value);

/** Reads stored or unknown input field by field; anything unusable takes the built-in default. */
export const normalizeImageExportSettings = (value: unknown): ImageExportSettings => {
  const record: Record<string, unknown> =
    typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  const { format, longEdge, metadata, namePattern, quality } = record;
  const roundedLongEdge = typeof longEdge === 'number' && Number.isFinite(longEdge) ? Math.round(longEdge) : 0;

  return {
    format: isOneOf(IMAGE_EXPORT_FORMATS, format) ? format : DEFAULT_IMAGE_EXPORT_SETTINGS.format,
    longEdge: roundedLongEdge >= 1 ? roundedLongEdge : null,
    metadata: isOneOf(IMAGE_EXPORT_METADATA, metadata) ? metadata : DEFAULT_IMAGE_EXPORT_SETTINGS.metadata,
    namePattern:
      typeof namePattern === 'string' &&
      namePattern.trim() !== '' &&
      namePattern.length <= IMAGE_EXPORT_NAME_PATTERN_MAX_LENGTH
        ? namePattern
        : DEFAULT_IMAGE_EXPORT_SETTINGS.namePattern,
    quality:
      typeof quality === 'number' && Number.isFinite(quality)
        ? Math.min(100, Math.max(1, Math.round(quality)))
        : DEFAULT_IMAGE_EXPORT_SETTINGS.quality,
  };
};

export interface ImageSize {
  width: number;
  height: number;
}

/** Python's `round`, which the server sizes with, rounds halves to even. */
const roundHalfEven = (value: number): number => {
  const floor = Math.floor(value);
  const fraction = value - floor;

  return fraction > 0.5 || (fraction === 0.5 && floor % 2 === 1) ? floor + 1 : floor;
};

/** The server's output size (`_target_size`): the longer side capped at `longEdge`, never enlarged, each side >= 1. */
export const getImageExportDimensions = (source: ImageSize, longEdge: number | null): ImageSize => {
  const longest = Math.max(source.width, source.height);

  if (longEdge === null || longEdge >= longest) {
    return { height: source.height, width: source.width };
  }

  const scale = longEdge / longest;

  return {
    height: Math.max(1, roundHalfEven(source.height * scale)),
    width: Math.max(1, roundHalfEven(source.width * scale)),
  };
};

/** The cap that scales the image to `scale` of its size; null at or above the original. */
export const getImageExportLongEdgeForScale = (source: ImageSize, scale: number): number | null =>
  scale >= 1 ? null : Math.max(1, Math.round(Math.max(source.width, source.height) * scale));

/** The cap that makes one side `length` px; null at or above that side's original length. */
export const getImageExportLongEdgeForSide = (
  source: ImageSize,
  side: keyof ImageSize,
  length: number
): number | null => getImageExportLongEdgeForScale(source, length / source[side]);

/** What the encoder reads: equal outputs share one request, whatever the settings they ignore hold. */
export interface ImageExportEncoding {
  format: ImageExportFormat;
  longEdge: number | null;
  metadata: ImageExportMetadata | null;
  quality: number | null;
}

/** With the source's size, a cap at or above its longer side is the original size. */
export const getImageExportEncoding = (settings: ImageExportSettings, source?: ImageSize): ImageExportEncoding => ({
  format: settings.format,
  longEdge:
    settings.longEdge !== null && source && settings.longEdge >= Math.max(source.width, source.height)
      ? null
      : settings.longEdge,
  metadata: settings.format === 'png' ? settings.metadata : null,
  quality: isLossyImageExportFormat(settings.format) ? settings.quality : null,
});

/** "WebP 85 · 1024 px": the format, its quality when lossy, then the size cap when one is set. */
export const getImageExportSummary = (
  settings: ImageExportSettings,
  formatLongEdge: (pixels: number) => string
): string => {
  const format = IMAGE_EXPORT_FORMAT_LABELS[settings.format];
  const encoding = isLossyImageExportFormat(settings.format) ? `${format} ${String(settings.quality)}` : format;

  return settings.longEdge === null ? encoding : `${encoding} · ${formatLongEdge(settings.longEdge)}`;
};
