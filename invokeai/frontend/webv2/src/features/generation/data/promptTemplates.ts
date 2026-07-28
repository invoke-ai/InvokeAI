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
import { absolutizeApiUrl, apiFetch, apiFetchJson } from '@platform/transport/http';
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
  /** Absolute URL of the preview image, or null when the template has none. */
  imageUrl: string | null;
}

export interface PromptTemplateDraft {
  name: string;
  negativePrompt: string;
  positivePrompt: string;
  /**
   * The image the template should end up with, always stated outright. The
   * backend replaces the whole record, so an absent image part *deletes* the
   * stored one — there is no "leave it as it was". The editor therefore loads
   * the current image up front with `fetchPromptTemplateImage`.
   */
  image: Blob | null;
}

export const promptTemplateKeys = {
  all: ['generation', 'promptTemplates'] as const,
};

const mapPromptTemplate = (dto: PromptTemplateDTO): PromptTemplateRecord => ({
  id: dto.id,
  imageUrl: dto.image ? absolutizeApiUrl(dto.image) : null,
  isDefault: dto.type === 'default',
  name: dto.name,
  negativePrompt: dto.preset_data.negative_prompt,
  positivePrompt: dto.preset_data.positive_prompt,
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
      queryKey: promptTemplateKeys.all,
      staleTime: 30_000,
    });
  })();

/**
 * Authored templates are always the user's own and private. The backend also
 * models admin-owned `default` templates and an `is_public` share flag; neither
 * is authored from here, so both are pinned rather than exposed.
 */
const toFormData = (draft: PromptTemplateDraft): FormData => {
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

  if (draft.image) {
    body.append('image', draft.image);
  }

  return body;
};

export const createPromptTemplate = async (draft: PromptTemplateDraft): Promise<PromptTemplateRecord> =>
  mapPromptTemplate(
    await apiFetchJson<PromptTemplateDTO>(`${PROMPT_TEMPLATES_BASE}/`, { body: toFormData(draft), method: 'POST' })
  );

export const updatePromptTemplate = async (id: string, draft: PromptTemplateDraft): Promise<PromptTemplateRecord> =>
  mapPromptTemplate(
    await apiFetchJson<PromptTemplateDTO>(`${PROMPT_TEMPLATES_BASE}/i/${encodeURIComponent(id)}`, {
      body: toFormData(draft),
      method: 'PATCH',
    })
  );

/**
 * Reads a template's existing preview image back as a blob so an edit that does
 * not touch the image can resend it. Returns null if it cannot be read — losing
 * the image is a smaller failure than blocking the edit.
 */
export const fetchPromptTemplateImage = async (imageUrl: string): Promise<Blob | null> => {
  try {
    const response = await apiFetch(imageUrl);

    return await response.blob();
  } catch {
    return null;
  }
};

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

export const invalidatePromptTemplates = async (queryClient: QueryClient): Promise<void> => {
  await queryClient.invalidateQueries({ queryKey: promptTemplateKeys.all });
};
