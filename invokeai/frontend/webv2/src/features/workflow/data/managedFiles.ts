import type { components } from '@api/schema';

import { apiFetch, apiFetchJson } from '@platform/transport/http';

type FileDTO = components['schemas']['FileDTO'];

/** A managed file as the workflow UI reads it; wire names stay in this module. */
export interface ManagedFile {
  id: string;
  name: string;
  contentType: string;
  sizeBytes: number;
  createdAt: string;
}

export const MANAGED_FILE_ACCEPTED_EXTENSIONS = ['.csv', '.json', '.md', '.markdown', '.pdf', '.txt', '.yaml', '.yml'];

/** The hidden input's `accept` value; the server remains the authority on what it stores. */
export const MANAGED_FILE_ACCEPT = MANAGED_FILE_ACCEPTED_EXTENSIONS.join(',');

const mapManagedFile = (file: FileDTO): ManagedFile => ({
  contentType: file.content_type,
  createdAt: file.created_at,
  id: file.file_id,
  name: file.file_name,
  sizeBytes: file.size_bytes,
});

export const managedFileQueryKey = (fileId: string) => ['workflow', 'managed-file', fileId] as const;

export const getManagedFile = async (fileId: string, signal?: AbortSignal): Promise<ManagedFile> => {
  const file = await apiFetchJson<FileDTO>(`/api/v1/files/i/${encodeURIComponent(fileId)}`, { signal });

  return mapManagedFile(file);
};

const MANAGED_FILE_STALE_TIME_MS = 5 * 60 * 1000;

/** Metadata for one uploaded file; an empty id leaves the query disabled. */
export const managedFileQueryOptions = (fileId: string) => ({
  enabled: fileId !== '',
  queryFn: ({ signal }: { signal: AbortSignal }): Promise<ManagedFile> => getManagedFile(fileId, signal),
  queryKey: managedFileQueryKey(fileId),
  retry: false,
  staleTime: MANAGED_FILE_STALE_TIME_MS,
});

export const uploadManagedFile = async (file: File, signal?: AbortSignal): Promise<ManagedFile> => {
  const body = new FormData();

  body.append('file', file);

  const uploaded = await apiFetchJson<FileDTO>('/api/v1/files/upload', { body, method: 'POST', signal });

  return mapManagedFile(uploaded);
};

/** Deleting a file is a server-side action; callers own when a reference no longer needs it. */
export const deleteManagedFile = async (fileId: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`/api/v1/files/i/${encodeURIComponent(fileId)}`, { method: 'DELETE', signal });
};
