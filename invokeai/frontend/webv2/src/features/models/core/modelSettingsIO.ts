import type { ModelConfig } from './types';

/**
 * Pure logic for exporting a model's user-editable settings to JSON and
 * validating/partitioning an import. The file format is shared with the
 * legacy frontend: `{name?, description?, source_url?, default_settings?,
 * trigger_phrases?, cpu_only?, cover_image? (data URL)}`. Anything that
 * fetches or writes lives in the UI layer.
 */

export const sanitizeFilename = (name: string): string => name.replace(/[<>:"/\\|?*]/g, '_');

export const isSafeUrl = (url: string): boolean => url.startsWith('https://') || url.startsWith('http://');

export const isImageDataUrl = (value: string): boolean => /^data:image\/[a-zA-Z0-9.+-]+;base64,/.test(value);

export const dataUrlToFile = (dataUrl: string, filename: string): File | null => {
  const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/);

  if (!match) {
    return null;
  }

  const mime = match[1];
  const b64 = match[2];

  if (!mime || !b64) {
    return null;
  }

  try {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    return new File([bytes], filename, { type: mime });
  } catch {
    return null;
  }
};

const IMPORTABLE_FIELDS = [
  'name',
  'description',
  'source_url',
  'default_settings',
  'trigger_phrases',
  'cpu_only',
] as const;

export const buildExportData = (model: ModelConfig): Record<string, unknown> => {
  const data: Record<string, unknown> = {};

  if (typeof model.name === 'string' && model.name.length > 0) {
    data.name = model.name;
  }

  if (typeof model.description === 'string' && model.description.length > 0) {
    data.description = model.description;
  }

  if (typeof model.source_url === 'string' && model.source_url.length > 0) {
    data.source_url = model.source_url;
  }

  if (model.default_settings !== undefined && model.default_settings !== null) {
    data.default_settings = model.default_settings;
  }

  if (model.trigger_phrases !== undefined && model.trigger_phrases !== null) {
    data.trigger_phrases = model.trigger_phrases;
  }

  if (model.cpu_only !== undefined && model.cpu_only !== null) {
    data.cpu_only = model.cpu_only;
  }

  return data;
};

export const validateImportData = (data: unknown): data is Record<string, unknown> => {
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    return false;
  }

  const obj = data as Record<string, unknown>;

  if (obj.name !== undefined && obj.name !== null && typeof obj.name !== 'string') {
    return false;
  }

  if (obj.description !== undefined && obj.description !== null && typeof obj.description !== 'string') {
    return false;
  }

  if (obj.source_url !== undefined && obj.source_url !== null) {
    if (typeof obj.source_url !== 'string') {
      return false;
    }

    if (obj.source_url.length > 0 && !isSafeUrl(obj.source_url)) {
      return false;
    }
  }

  if (obj.cover_image !== undefined && obj.cover_image !== null) {
    if (typeof obj.cover_image !== 'string' || !isImageDataUrl(obj.cover_image)) {
      return false;
    }
  }

  if (obj.trigger_phrases !== undefined) {
    if (!Array.isArray(obj.trigger_phrases) || !obj.trigger_phrases.every((phrase) => typeof phrase === 'string')) {
      return false;
    }
  }

  if (obj.default_settings !== undefined) {
    if (
      typeof obj.default_settings !== 'object' ||
      obj.default_settings === null ||
      Array.isArray(obj.default_settings)
    ) {
      return false;
    }
  }

  if (obj.cpu_only !== undefined && typeof obj.cpu_only !== 'boolean') {
    return false;
  }

  return true;
};

/**
 * Split validated import data into a PATCH body of fields the target model's
 * config actually carries, and the fields skipped as incompatible (e.g.
 * cpu_only imported onto a main model). The cover image is not part of the
 * body — it uploads through its own endpoint.
 */
export const partitionImportableFields = (
  data: Record<string, unknown>,
  model: ModelConfig
): { body: Record<string, unknown>; skipped: string[] } => {
  const body: Record<string, unknown> = {};
  const skipped: string[] = [];

  for (const field of IMPORTABLE_FIELDS) {
    if (data[field] === undefined || data[field] === null) {
      continue;
    }

    if (field in model) {
      body[field] = data[field];
    } else {
      skipped.push(field);
    }
  }

  return { body, skipped };
};
