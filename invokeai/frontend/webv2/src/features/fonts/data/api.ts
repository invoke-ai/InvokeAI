import type {
  FontAxis,
  FontCatalogPage,
  FontDownloadReference,
  FontInstance,
  FontRecord,
  FontScope,
} from '@features/fonts/core/types';

import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { apiFetch, apiFetchJson } from '@platform/transport/http';

import type { FontListParams } from './keys';

export type { FontListParams } from './keys';

const FONTS_BASE = '/api/v1/fonts';

interface FontAxisDTO {
  tag: string;
  label?: string;
  minimum: number;
  default: number;
  maximum: number;
  hidden?: boolean;
}

interface FontInstanceDTO {
  name: string;
  coordinates: Record<string, number>;
}

interface FontRecordDTO {
  id: string;
  family: string;
  label: string;
  style: string;
  weight: number;
  content_hash: string;
  scope: FontScope;
  source: 'uploaded' | 'directory';
  filename: string;
  byte_size: number;
  url: string;
  axes?: FontAxisDTO[];
  instances?: FontInstanceDTO[];
}

interface FontCatalogPageDTO {
  items: FontRecordDTO[];
  total: number;
  offset: number;
  limit: number;
}

interface FontValidationDTO {
  filename: string;
  family: string;
  label: string;
  style: string;
  weight: number;
  content_hash: string;
  byte_size: number;
  axes?: FontAxisDTO[];
  instances?: FontInstanceDTO[];
}

export interface FontRescanResult {
  indexed: number;
  revision: number;
}

export interface FontValidationResult {
  filename: string;
  family: string;
  label: string;
  style: string;
  weight: number;
  contentHash: string;
  byteSize: number;
  axes: readonly FontAxis[];
  instances: readonly FontInstance[];
}

export interface UploadFontResult {
  font: FontRecord;
  created: boolean;
}

const mapAxis = (axis: FontAxisDTO): FontAxis => ({
  default: axis.default,
  hidden: axis.hidden === true,
  label: axis.label?.trim() || axis.tag,
  maximum: axis.maximum,
  minimum: axis.minimum,
  tag: axis.tag,
});

const mapInstance = (instance: FontInstanceDTO): FontInstance => ({
  coordinates: { ...instance.coordinates },
  name: instance.name,
});

export const mapFontRecord = (dto: FontRecordDTO): FontRecord => ({
  axes: (dto.axes ?? []).map(mapAxis),
  byteSize: dto.byte_size,
  contentHash: dto.content_hash,
  family: dto.family,
  filename: dto.filename,
  id: dto.id,
  instances: (dto.instances ?? []).map(mapInstance),
  label: dto.label,
  scope: dto.scope,
  source: dto.source,
  style: dto.style,
  url: dto.url,
  weight: dto.weight,
});

const mapValidationResult = (dto: FontValidationDTO): FontValidationResult => ({
  axes: (dto.axes ?? []).map(mapAxis),
  byteSize: dto.byte_size,
  contentHash: dto.content_hash,
  family: dto.family,
  filename: dto.filename,
  instances: (dto.instances ?? []).map(mapInstance),
  label: dto.label,
  style: dto.style,
  weight: dto.weight,
});

const buildListUrl = ({
  contentHash,
  limit = 100,
  offset = 0,
  scope = 'all',
  search = '',
}: FontListParams = {}): string => {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });

  if (contentHash) {
    params.set('content_hash', contentHash);
  }
  if (search.trim()) {
    params.set('search', search.trim());
  }
  if (scope !== 'all') {
    params.set('scope', scope);
  }

  return `${FONTS_BASE}?${params.toString()}`;
};

export const listFonts = async (params: FontListParams = {}, signal?: AbortSignal): Promise<FontCatalogPage> => {
  const dto = await apiFetchJson<FontCatalogPageDTO>(buildListUrl(params), { signal });
  const items = dto.items.map(mapFontRecord);

  return {
    items,
    limit: dto.limit,
    offset: dto.offset,
    total: dto.total,
  };
};

export const getFont = async (id: string, signal?: AbortSignal): Promise<FontRecord> => {
  const dto = await apiFetchJson<FontRecordDTO>(`${FONTS_BASE}/${encodeURIComponent(id)}`, { signal });

  return mapFontRecord(dto);
};

const getFontFileUrl = (reference: FontDownloadReference): string => {
  const params = new URLSearchParams();

  if (reference.contentHash) {
    params.set('expected_hash', reference.contentHash);
  }

  const query = params.toString();
  return `${FONTS_BASE}/${encodeURIComponent(reference.id)}/file${query ? `?${query}` : ''}`;
};

/** Downloads an original face or a pinned static variation instance. */
export const downloadFont = async (reference: FontDownloadReference, signal?: AbortSignal): Promise<Uint8Array> => {
  const owner = captureAccountScope();
  const hasAxes = reference.axes !== undefined && Object.keys(reference.axes).length > 0;
  const response = hasAxes
    ? await apiFetch(`${FONTS_BASE}/${encodeURIComponent(reference.id)}/instance`, {
        body: JSON.stringify({
          content_hash: reference.contentHash ?? undefined,
          coordinates: reference.axes,
        }),
        headers: { 'Content-Type': 'application/json' },
        method: 'POST',
        signal,
      })
    : await apiFetch(getFontFileUrl(reference), { signal });

  const bytes = new Uint8Array(await response.arrayBuffer());
  assertAccountScopeCurrent(owner);
  return bytes;
};

/** Validates and extracts metadata without publishing the upload. */
export const validateFont = async (file: Blob, signal?: AbortSignal): Promise<FontValidationResult> => {
  const body = new FormData();
  body.append('file', file);
  const dto = await apiFetchJson<FontValidationDTO>(`${FONTS_BASE}/validate`, { body, method: 'POST', signal });

  return mapValidationResult(dto);
};

export const uploadFont = async (
  file: Blob,
  scope: FontScope = 'private',
  signal?: AbortSignal
): Promise<UploadFontResult> => {
  const body = new FormData();
  body.append('file', file);
  body.append('scope', scope);
  const dto = await apiFetchJson<{ font: FontRecordDTO; created: boolean }>(FONTS_BASE, {
    body,
    method: 'POST',
    signal,
  });

  return { created: dto.created, font: mapFontRecord(dto.font) };
};

export const deleteFont = async (id: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`${FONTS_BASE}/${encodeURIComponent(id)}`, { method: 'DELETE', signal });
};

export const rescanFonts = (signal?: AbortSignal): Promise<FontRescanResult> =>
  apiFetchJson<FontRescanResult>(`${FONTS_BASE}/rescan`, { method: 'POST', signal });
