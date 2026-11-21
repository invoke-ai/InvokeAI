export const IMAGE_EXPORT_FORMATS = ['png', 'jpeg', 'webp', 'avif'] as const;

export type ImageExportFormat = (typeof IMAGE_EXPORT_FORMATS)[number];

export const IMAGE_EXPORT_METADATA = ['all', 'parameters', 'none'] as const;

export type ImageExportMetadata = (typeof IMAGE_EXPORT_METADATA)[number];

export interface ImageExportSettings {
  format: ImageExportFormat;
  quality: number;
  longEdge: number | null;
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

export const getImageExportLongEdgeForScale = (source: ImageSize, scale: number): number | null =>
  scale >= 1 ? null : Math.max(1, Math.round(Math.max(source.width, source.height) * scale));

export const getImageExportLongEdgeForSide = (
  source: ImageSize,
  side: keyof ImageSize,
  length: number
): number | null => getImageExportLongEdgeForScale(source, length / source[side]);

/** Only what the encoder reads, so equal outputs share one request. */
export interface ImageExportEncoding {
  format: ImageExportFormat;
  longEdge: number | null;
  metadata: ImageExportMetadata | null;
  quality: number | null;
}

export const getImageExportEncoding = (settings: ImageExportSettings, source?: ImageSize): ImageExportEncoding => ({
  format: settings.format,
  longEdge:
    settings.longEdge !== null && source && settings.longEdge >= Math.max(source.width, source.height)
      ? null
      : settings.longEdge,
  metadata: settings.format === 'png' ? settings.metadata : null,
  quality: isLossyImageExportFormat(settings.format) ? settings.quality : null,
});
