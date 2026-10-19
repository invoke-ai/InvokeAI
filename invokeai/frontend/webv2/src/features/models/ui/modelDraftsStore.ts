import type { ModelEditFormValues } from '@features/models/core/schemas';
import type { ModelConfig } from '@features/models/core/types';

import { getModelsSnapshot, subscribeModels } from '@features/models/data/modelsStore';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createKeyedTransientStore } from '@platform/state/externalStore';
import { useExternalStoreSelector, type EqualityFn } from '@platform/state/selectors';
import { useCallback } from 'react';

/**
 * Unsaved model-detail edits, kept per model while the manager navigates between tabs and models so leaving a form
 * never loses work. Memory only, for the account's lifetime in this tab; a draft ends on its surface's save or
 * discard, the model's removal, eviction past the bound, or an account transition. A save in flight belongs here too,
 * so a form remounted mid-request still shows it and its outcome.
 */

type DraftValues = Readonly<Record<string, unknown>>;

// Form value types are interfaces without index signatures, so they enter as plain objects.
const readField = (values: object, key: string): unknown => (values as DraftValues)[key];

export interface ModelFieldsDraft {
  /** The user's value for each field that differs from the server record (or from a save in flight). */
  readonly fields: DraftValues;
  /** The server value each of those fields was edited against; a later server change to it is a conflict. */
  readonly baseline: DraftValues;
  /** The surface's full values while a save of them is in flight. */
  readonly submitted?: DraftValues;
}

export interface ModelIdentityDraft extends ModelFieldsDraft {
  /** The last failed save's message, shown until the next save attempt. */
  readonly error: string | null;
}

export interface ModelDraft {
  /** Default-settings edits; absent when nothing differs from the server and no save is in flight. */
  readonly defaults?: ModelFieldsDraft;
  /** Present exactly while the identity editor is open, even before anything is edited. */
  readonly identity?: ModelIdentityDraft;
}

export type ModelDraftSurface = keyof ModelDraft;

/** Drafts with unsaved changes beyond this many evict the least recently edited; open editors are bounded apart. */
export const MAX_MODEL_DRAFTS = 20;

const EMPTY_FIELDS_DRAFT: ModelFieldsDraft = { baseline: {}, fields: {} };
const EMPTY_IDENTITY_DRAFT: ModelIdentityDraft = { ...EMPTY_FIELDS_DRAFT, error: null };

type ModelIdentitySource = Pick<
  ModelConfig,
  'base' | 'config_path' | 'description' | 'format' | 'name' | 'prediction_type' | 'source_url' | 'type' | 'variant'
>;

/** The identity editor's values for a server record; drafts of that surface are differences from these. */
export const toModelIdentityValues = (model: ModelIdentitySource): ModelEditFormValues => ({
  base: String(model.base),
  configPath: model.config_path ?? '',
  description: model.description ?? '',
  format: String(model.format),
  name: model.name,
  predictionType: (model.prediction_type ?? '') as ModelEditFormValues['predictionType'],
  sourceUrl: model.source_url ?? '',
  type: String(model.type),
  variant: model.variant ?? '',
});

const SERVER_VALUES: Record<ModelDraftSurface, (model: ModelConfig) => object> = {
  defaults: (model) => model.default_settings ?? {},
  identity: toModelIdentityValues,
};

// Unset and null are the same "no value" to the backend and to every form here.
const isSameValue = (left: unknown, right: unknown): boolean => Object.is(left ?? null, right ?? null);

export const hasDraftFields = (draft: ModelFieldsDraft | undefined): boolean =>
  draft !== undefined && Object.keys(draft.fields).length > 0;

/** The form's values: the current server record with the user's edited fields on top. */
export const applyModelDraftFields = <Values extends object>(
  server: Values,
  draft: ModelFieldsDraft | undefined
): Values => (draft !== undefined && hasDraftFields(draft) ? { ...server, ...draft.fields } : server);

/** True when the server changed a field the user also edited, to something other than the user's value. */
export const hasModelDraftConflict = (server: object, draft: ModelFieldsDraft | undefined): boolean =>
  draft !== undefined &&
  Object.keys(draft.fields).some(
    (key) =>
      !isSameValue(draft.baseline[key], readField(server, key)) &&
      !isSameValue(draft.fields[key], readField(server, key))
  );

export const hasUnsavedModelChanges = (draft: ModelDraft | undefined): boolean =>
  hasDraftFields(draft?.identity) || hasDraftFields(draft?.defaults);

/**
 * Whether a user value is still an edit: it differs from the server, or from a save in flight (a field typed back
 * to the old server value mid-save must survive that save landing).
 */
const isEdit = (value: unknown, key: string, server: object, submitted: DraftValues | undefined): boolean =>
  !isSameValue(value, readField(server, key)) || (submitted !== undefined && !isSameValue(value, submitted[key]));

const store = createKeyedTransientStore<string, ModelDraft>();
/** Draft keys from least to most recently edited; an entry joins at the end and moves only when edited. */
const recency = new Set<string>();

const deleteDraft = (modelKey: string): void => {
  recency.delete(modelKey);
  store.delete(modelKey);
};

/** Dirty drafts and clean open editors are bounded separately, so opening editors never evicts unsaved work. */
const evictBeyondBound = (): void => {
  for (const dirty of [true, false]) {
    const keys = [...recency].filter((key) => hasUnsavedModelChanges(store.get(key)) === dirty);

    for (const key of keys.slice(0, Math.max(0, keys.length - MAX_MODEL_DRAFTS))) {
      deleteDraft(key);
    }
  }
};

const writeDraft = (modelKey: string, draft: ModelDraft, edited = false): void => {
  if (draft.identity === undefined && draft.defaults === undefined) {
    deleteDraft(modelKey);
    return;
  }

  if (edited) {
    recency.delete(modelKey);
  }

  recency.add(modelKey);
  store.set(modelKey, draft);
  evictBeyondBound();
};

const withSurface = (
  draft: ModelDraft | undefined,
  surface: ModelDraftSurface,
  value: ModelFieldsDraft | ModelIdentityDraft | undefined
): ModelDraft => {
  const { [surface]: _replaced, ...rest } = draft ?? {};

  return value === undefined ? rest : { ...rest, [surface]: value };
};

/** Identity stays while its editor is open; defaults stay only while they hold edits or a save. */
const withFields = (
  draft: ModelDraft | undefined,
  surface: ModelDraftSurface,
  next: ModelFieldsDraft,
  error: string | null = draft?.identity?.error ?? null
): ModelDraft => {
  if (surface === 'identity') {
    return withSurface(draft, surface, { ...next, error });
  }

  return withSurface(draft, surface, hasDraftFields(next) || next.submitted !== undefined ? next : undefined);
};

/** Keep only fields that are still edits against `server`; unchanged input keeps its identity. */
const retainEdits = (previous: ModelFieldsDraft, server: object): ModelFieldsDraft => {
  const kept = Object.keys(previous.fields).filter((key) =>
    isEdit(previous.fields[key], key, server, previous.submitted)
  );

  if (kept.length === Object.keys(previous.fields).length) {
    return previous;
  }

  return {
    ...previous,
    baseline: Object.fromEntries(kept.map((key) => [key, previous.baseline[key]])),
    fields: Object.fromEntries(kept.map((key) => [key, previous.fields[key]])),
  };
};

let lastLoadedModels: ReadonlyMap<string, ModelConfig> | null = null;

registerAccountOwnedResource({
  clear: () => {
    recency.clear();
    store.clear();
    lastLoadedModels = null;
  },
  name: 'model-drafts',
});

/**
 * Follow the loaded library: drop drafts of models that leave it (deleted here or found gone by a refresh), and when
 * a record changes, drop edits the server now agrees with so nothing reads as unsaved without a difference.
 */
const syncWithLibrary = (): void => {
  const { modelsByKey, status } = getModelsSnapshot();

  if (status !== 'loaded' || modelsByKey === lastLoadedModels) {
    return;
  }

  const previousModels = lastLoadedModels;

  lastLoadedModels = modelsByKey;

  if (previousModels === null) {
    return;
  }

  // Deleting visited or later keys mid-iteration is safe for a Set; rewrites never re-add a key.
  for (const key of recency) {
    const model = modelsByKey.get(key);
    const previousModel = previousModels.get(key);

    if (previousModel === undefined || model === previousModel) {
      continue;
    }

    if (model === undefined) {
      deleteDraft(key);
      continue;
    }

    let draft = store.get(key);

    for (const surface of ['identity', 'defaults'] as const) {
      const surfaceDraft = draft?.[surface];

      if (surfaceDraft !== undefined) {
        const next = retainEdits(surfaceDraft, SERVER_VALUES[surface](model));

        if (next !== surfaceDraft) {
          draft = withFields(draft, surface, next);
        }
      }
    }

    if (draft !== store.get(key)) {
      writeDraft(key, draft ?? {});
    }
  }
};

subscribeModels(syncWithLibrary);
// The library may already be loaded when this lazy module first evaluates.
syncWithLibrary();

export const getModelDraft = (modelKey: string): ModelDraft | undefined => store.get(modelKey);

export const openModelIdentityDraft = (modelKey: string): void => {
  const draft = store.get(modelKey);

  if (draft?.identity === undefined) {
    writeDraft(modelKey, withSurface(draft, 'identity', EMPTY_IDENTITY_DRAFT));
  }
};

/**
 * Record a surface's full current values against the server record it was edited from. Only edits are kept; a field
 * keeps the baseline it was first edited against until it stops being an edit.
 */
export const recordModelDraftFields = (
  modelKey: string,
  surface: ModelDraftSurface,
  values: object,
  server: object
): void => {
  const draft = store.get(modelKey);
  const previous = draft?.[surface];
  const submitted = previous?.submitted;
  const fields: Record<string, unknown> = {};
  const baseline: Record<string, unknown> = {};

  for (const key of new Set([...Object.keys(server), ...Object.keys(values)])) {
    if (isEdit(readField(values, key), key, server, submitted)) {
      fields[key] = readField(values, key);
      baseline[key] =
        previous !== undefined && Object.hasOwn(previous.fields, key) ? previous.baseline[key] : readField(server, key);
    }
  }

  writeDraft(modelKey, withFields(draft, surface, { ...previous, baseline, fields }), true);
};

/** Mark a surface's save in flight; false while one already is, so a remounted form cannot send a second. */
export const beginModelDraftSave = (modelKey: string, surface: ModelDraftSurface, submitted: object): boolean => {
  const draft = store.get(modelKey);
  const previous = draft?.[surface] ?? EMPTY_FIELDS_DRAFT;

  if (previous.submitted !== undefined) {
    return false;
  }

  writeDraft(modelKey, withFields(draft, surface, { ...previous, submitted: submitted as DraftValues }, null));

  return true;
};

/**
 * A save landed: drop the fields it carried and keep anything edited since, as a draft against the saved record.
 * An identity editor left with nothing to save closes.
 */
export const finishModelDraftSave = (modelKey: string, surface: ModelDraftSurface, saved: object): void => {
  const draft = store.get(modelKey);
  const previous = draft?.[surface];
  const submitted = previous?.submitted;

  if (previous === undefined || submitted === undefined) {
    return;
  }

  // Equal to what was sent means saved; anything else was typed after the click.
  const kept = Object.keys(previous.fields).filter(
    (key) => isEdit(previous.fields[key], key, saved, undefined) && !isSameValue(previous.fields[key], submitted[key])
  );
  const next: ModelFieldsDraft = {
    baseline: Object.fromEntries(kept.map((key) => [key, readField(saved, key)])),
    fields: Object.fromEntries(kept.map((key) => [key, previous.fields[key]])),
  };

  writeDraft(
    modelKey,
    surface === 'identity' && !hasDraftFields(next)
      ? withSurface(draft, surface, undefined)
      : withFields(draft, surface, next, null)
  );
};

/** A save failed: the draft stays with the error. A draft discarded or evicted meanwhile is not revived. */
export const failModelDraftSave = (
  modelKey: string,
  surface: ModelDraftSurface,
  server: object,
  error: string | null
): void => {
  const draft = store.get(modelKey);
  const previous = draft?.[surface];

  if (previous?.submitted === undefined) {
    return;
  }

  const { submitted: _settled, ...settled } = previous;

  writeDraft(modelKey, withFields(draft, surface, retainEdits(settled, server), error));
};

/** A new attempt replaces the last failure's message. */
export const clearModelIdentitySaveError = (modelKey: string): void => {
  const draft = store.get(modelKey);

  if (draft?.identity !== undefined && draft.identity.error !== null) {
    writeDraft(modelKey, withSurface(draft, 'identity', { ...draft.identity, error: null }));
  }
};

/** The intentional discard: Cancel/Reset. */
export const discardModelDraft = (modelKey: string, surface: ModelDraftSurface): void => {
  const draft = store.get(modelKey);

  if (draft?.[surface] !== undefined) {
    writeDraft(modelKey, withSurface(draft, surface, undefined));
  }
};

/** Subscribes to one model's draft only, so edits elsewhere never re-render this caller. */
export const useModelDraft = <Selected>(
  modelKey: string,
  selector: (draft: ModelDraft | undefined) => Selected,
  isEqual: EqualityFn<Selected> = Object.is
): Selected => {
  const subscribe = useCallback((listener: () => void) => store.subscribeKey(modelKey, listener), [modelKey]);
  const getSnapshot = useCallback(() => store.get(modelKey), [modelKey]);

  return useExternalStoreSelector(subscribe, getSnapshot, selector, isEqual);
};

export const useModelHasUnsavedChanges = (modelKey: string): boolean => useModelDraft(modelKey, hasUnsavedModelChanges);
