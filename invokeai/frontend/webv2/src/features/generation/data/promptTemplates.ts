/**
 * The current user's prompt templates, plus the catalog shipped with the backend.
 *
 * The route is `/api/v1/style_presets/` — "style preset" is the backend's name
 * for the same thing, kept for wire compatibility with the legacy client, which
 * writes to and reads from this exact store. Only this module speaks that
 * vocabulary; everything above it says "prompt template".
 *
 * Create and update are multipart because a template may carry a preview image.
 * `apiFetchJson` leaves `FormData` bodies alone, so the browser sets its own
 * boundary while the injected `Authorization` header still rides along.
 */

import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { QueryClient } from '@tanstack/react-query';

import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { apiFetch, apiFetchJson } from '@platform/transport/http';
import { queryOptions } from '@tanstack/react-query';

const PROMPT_TEMPLATES_BASE = '/api/v1/style_presets';

/** Private snake_case wire shape; never crosses the feature interface. */
interface PromptTemplateDTO {
  id: string;
  name: string;
  type: 'default' | 'user';
  is_public: boolean;
  user_id: string;
  image: string | null;
  preset_data: { positive_prompt: string; negative_prompt: string };
}

export interface PromptTemplateRecord extends PromptTemplateSnapshot {
  /** `default` templates ship with the backend and are read-only here. */
  isDefault: boolean;
  /** Whether the authenticated image resource should be requested. */
  hasImage: boolean;
  isPublic: boolean;
  userId: string;
}

/**
 * The four fields a record shares with a snapshot, and no more.
 *
 * A record is assignable to a snapshot, so applying one type-checks and keeps
 * catalog-only fields such as `isDefault` and `hasImage` along for the ride.
 * They then land in persisted project state and every queue item's widget
 * snapshot. Extra keys also make `isCanonicalPromptTemplateSnapshot` reject the
 * object, so the identity short-circuits in settings normalization never fire.
 */
export const toPromptTemplateSnapshot = ({
  id,
  name,
  negativePrompt,
  positivePrompt,
}: PromptTemplateSnapshot): PromptTemplateSnapshot => ({ id, name, negativePrompt, positivePrompt });

interface PromptTemplateDraftFields {
  name: string;
  negativePrompt: string;
  positivePrompt: string;
}

export interface PromptTemplateCreateDraft extends PromptTemplateDraftFields {
  image: Blob | null;
}

export type PromptTemplateImageUpdate = { kind: 'preserve' } | { kind: 'remove' } | { blob: Blob; kind: 'replace' };

export interface PromptTemplateUpdateDraft extends PromptTemplateDraftFields {
  image: PromptTemplateImageUpdate;
}

export const promptTemplateKeys = {
  all: ['generation', 'promptTemplates'] as const,
  image: (id: string) => [...promptTemplateKeys.all, 'image', id] as const,
  list: () => [...promptTemplateKeys.all, 'list'] as const,
};

const mapPromptTemplate = (dto: PromptTemplateDTO): PromptTemplateRecord => ({
  hasImage: dto.image !== null,
  id: dto.id,
  isDefault: dto.type === 'default',
  isPublic: dto.is_public,
  name: dto.name,
  negativePrompt: dto.preset_data.negative_prompt,
  positivePrompt: dto.preset_data.positive_prompt,
  userId: dto.user_id,
});

export const promptTemplatesQueryOptions = () =>
  (() => {
    const owner = captureAccountScope();

    return queryOptions({
      queryFn: async ({ signal }): Promise<PromptTemplateRecord[]> => {
        const requestSignal = AbortSignal.any([signal, owner.signal]);
        const templates = await apiFetchJson<PromptTemplateDTO[]>(`${PROMPT_TEMPLATES_BASE}/`, {
          signal: requestSignal,
        });

        assertAccountScopeCurrent(owner);
        return templates.map(mapPromptTemplate);
      },
      queryKey: promptTemplateKeys.list(),
      staleTime: 30_000,
    });
  })();

/**
 * Authored templates are always the user's own and private. The backend also
 * models admin-owned `default` templates and an `is_public` share flag; neither
 * is authored from here, so both are pinned rather than exposed.
 */
const toFormData = (draft: PromptTemplateDraftFields): FormData => {
  const body = new FormData();

  body.append(
    'data',
    JSON.stringify({
      is_public: false,
      name: draft.name,
      negative_prompt: draft.negativePrompt,
      positive_prompt: draft.positivePrompt,
      type: 'user',
    })
  );

  return body;
};

export const createPromptTemplate = async (draft: PromptTemplateCreateDraft): Promise<PromptTemplateRecord> => {
  const body = toFormData(draft);

  if (draft.image) {
    body.append('image', draft.image);
  }

  return mapPromptTemplate(
    await apiFetchJson<PromptTemplateDTO>(`${PROMPT_TEMPLATES_BASE}/`, { body, method: 'POST' })
  );
};

export const updatePromptTemplate = async (
  id: string,
  draft: PromptTemplateUpdateDraft
): Promise<PromptTemplateRecord> => {
  const body = toFormData(draft);

  if (draft.image.kind === 'preserve') {
    body.append('preserve_image', 'true');
  } else if (draft.image.kind === 'replace') {
    body.append('image', draft.image.blob);
  }

  return mapPromptTemplate(
    await apiFetchJson<PromptTemplateDTO>(`${PROMPT_TEMPLATES_BASE}/i/${encodeURIComponent(id)}`, {
      body,
      method: 'PATCH',
    })
  );
};

export const fetchPromptTemplateImage = async (id: string, signal?: AbortSignal): Promise<Blob> => {
  const response = await apiFetch(`${PROMPT_TEMPLATES_BASE}/i/${encodeURIComponent(id)}/image`, { signal });

  return await response.blob();
};

export const promptTemplateImageQueryOptions = (id: string) =>
  (() => {
    const owner = captureAccountScope();

    return queryOptions({
      queryFn: ({ signal }): Promise<Blob> =>
        fetchPromptTemplateImage(id, AbortSignal.any([signal, owner.signal])).then((image) => {
          assertAccountScopeCurrent(owner);
          return image;
        }),
      queryKey: promptTemplateKeys.image(id),
      staleTime: Number.POSITIVE_INFINITY,
    });
  })();

export const deletePromptTemplate = async (id: string): Promise<void> => {
  await apiFetch(`${PROMPT_TEMPLATES_BASE}/i/${encodeURIComponent(id)}`, { method: 'DELETE' });
};

/** Accepts the CSV and JSON shapes documented by the backend's importer. */
export const importPromptTemplates = async (file: File): Promise<void> => {
  const body = new FormData();

  body.append('file', file);
  await apiFetch(`${PROMPT_TEMPLATES_BASE}/import`, { body, method: 'POST' });
};

/**
 * Returns CSV, not JSON, so this goes through `apiFetch` rather than
 * `apiFetchJson`. It also cannot be an anchor download: the URL alone carries no
 * `Authorization` header, and only the transport injects one.
 */
export const exportPromptTemplates = async (): Promise<Blob> => {
  const response = await apiFetch(`${PROMPT_TEMPLATES_BASE}/export`);

  return await response.blob();
};

export const invalidatePromptTemplates = async (queryClient: QueryClient, id?: string): Promise<void> => {
  const invalidations: Promise<void>[] = [
    queryClient.invalidateQueries({ queryKey: promptTemplateKeys.list(), exact: true }),
  ];

  if (id) {
    invalidations.push(queryClient.invalidateQueries({ queryKey: promptTemplateKeys.image(id), exact: true }));
  }

  await Promise.all(invalidations);
};
