import type { AnyModelConfigWithExternal } from 'services/api/types';

export const sanitizeFilename = (name: string): string => {
  return name.replace(/[<>:"/\\|?*]/g, '_');
};

export const isSafeUrl = (url: string): boolean => {
  return url.startsWith('https://') || url.startsWith('http://');
};

export const isImageDataUrl = (value: string): boolean => {
  return /^data:image\/[a-zA-Z0-9.+-]+;base64,/.test(value);
};

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

export const buildExportData = (modelConfig: AnyModelConfigWithExternal): Record<string, unknown> => {
  const data: Record<string, unknown> = {};

  if ('name' in modelConfig && typeof modelConfig.name === 'string' && modelConfig.name.length > 0) {
    data.name = modelConfig.name;
  }

  if (
    'description' in modelConfig &&
    typeof modelConfig.description === 'string' &&
    modelConfig.description.length > 0
  ) {
    data.description = modelConfig.description;
  }

  if ('source_url' in modelConfig && typeof modelConfig.source_url === 'string' && modelConfig.source_url.length > 0) {
    data.source_url = modelConfig.source_url;
  }

  if (
    'default_settings' in modelConfig &&
    modelConfig.default_settings !== undefined &&
    modelConfig.default_settings !== null
  ) {
    data.default_settings = modelConfig.default_settings;
  }

  if (
    'trigger_phrases' in modelConfig &&
    modelConfig.trigger_phrases !== undefined &&
    modelConfig.trigger_phrases !== null
  ) {
    data.trigger_phrases = modelConfig.trigger_phrases;
  }

  if ('cpu_only' in modelConfig && modelConfig.cpu_only !== null) {
    data.cpu_only = modelConfig.cpu_only;
  }

  return data;
};

export const fetchImageAsDataUrl = async (url: string): Promise<string | null> => {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    const blob = await response.blob();
    if (!blob.type.startsWith('image/')) {
      return null;
    }
    return await new Promise<string | null>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
};

export const validateImportData = (data: unknown): data is Record<string, unknown> => {
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    return false;
  }

  const obj = data as Record<string, unknown>;

  if ('name' in obj && obj.name !== undefined && obj.name !== null) {
    if (typeof obj.name !== 'string') {
      return false;
    }
  }

  if ('description' in obj && obj.description !== undefined && obj.description !== null) {
    if (typeof obj.description !== 'string') {
      return false;
    }
  }

  if ('source_url' in obj && obj.source_url !== undefined && obj.source_url !== null) {
    if (typeof obj.source_url !== 'string') {
      return false;
    }
    if (obj.source_url.length > 0 && !isSafeUrl(obj.source_url)) {
      return false;
    }
  }

  if ('cover_image' in obj && obj.cover_image !== undefined && obj.cover_image !== null) {
    if (typeof obj.cover_image !== 'string' || !isImageDataUrl(obj.cover_image)) {
      return false;
    }
  }

  if ('trigger_phrases' in obj && obj.trigger_phrases !== undefined) {
    if (!Array.isArray(obj.trigger_phrases) || !obj.trigger_phrases.every((p) => typeof p === 'string')) {
      return false;
    }
  }

  if ('default_settings' in obj && obj.default_settings !== undefined) {
    if (
      typeof obj.default_settings !== 'object' ||
      obj.default_settings === null ||
      Array.isArray(obj.default_settings)
    ) {
      return false;
    }
  }

  if ('cpu_only' in obj && obj.cpu_only !== undefined) {
    if (typeof obj.cpu_only !== 'boolean') {
      return false;
    }
  }

  return true;
};
