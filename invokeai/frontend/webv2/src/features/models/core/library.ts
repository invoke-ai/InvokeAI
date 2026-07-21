import type { ModelConfig, ModelTaxonomyType } from './types';

import { getModelBaseLabel, KNOWN_MODEL_BASES } from './baseIdentity';
import { getModelCategoryRank, getModelTypePluralLabel } from './taxonomy';

/**
 * Pure filtering/grouping/sorting for the model library. Kept free of React so
 * the list views stay thin and this logic is unit-testable.
 */

export type ModelSortField = 'default' | 'name' | 'base' | 'size' | 'format';

export interface ModelLibraryFilters {
  searchTerm: string;
  /** null = all types. */
  typeFilter: ModelTaxonomyType | null;
  /** null = all bases. */
  baseFilter: string | null;
  /** Only models whose files are missing on disk. */
  missingOnly: boolean;
  sortField: ModelSortField;
  sortDirection: 'asc' | 'desc';
}

export const DEFAULT_LIBRARY_FILTERS: ModelLibraryFilters = {
  baseFilter: null,
  missingOnly: false,
  searchTerm: '',
  sortDirection: 'asc',
  sortField: 'default',
  typeFilter: null,
};

export interface ModelGroup {
  type: ModelTaxonomyType;
  label: string;
  models: ModelConfig[];
}

export interface ModelPickerOptions {
  /** Multi-select base filter; empty/undefined = all bases. */
  baseFilter?: ReadonlySet<string>;
  /** Hide specific models, e.g. the current model or already-linked choices. */
  excludeKeys?: ReadonlySet<string>;
  /** Extra predicate supplied by the owning form, e.g. base compatibility. */
  filter?: (model: ModelConfig) => boolean;
  modelTypes: ModelTaxonomyType[];
  searchTerm: string;
}

/**
 * Picker list group: one per (type, base) pair so items never need a per-row
 * base badge — the header renders the base (and, for multi-type pickers, the
 * type label) from `base`/`type`.
 */
export interface ModelPickerGroup {
  key: string;
  type: ModelTaxonomyType;
  base: string;
  models: ModelConfig[];
}

export interface ModelPickerResult {
  /** Distinct bases among candidates (pre-search, pre-base-filter), for chips. */
  availableBases: string[];
  /** Models available before text search, used for empty-state copy. */
  candidates: ModelConfig[];
  groups: ModelPickerGroup[];
}

const matchesSearch = (model: ModelConfig, searchTerm: string): boolean => {
  if (!searchTerm) {
    return true;
  }

  const haystack =
    `${model.name} ${model.description ?? ''} ${model.base} ${model.type} ${model.format} ${(model.trigger_phrases ?? []).join(' ')}`.toLowerCase();

  return searchTerm
    .toLowerCase()
    .split(/\s+/)
    .every((term) => haystack.includes(term));
};

const compareBySortField = (a: ModelConfig, b: ModelConfig, field: ModelSortField): number => {
  switch (field) {
    case 'name':
    case 'default':
      return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
    case 'base':
      return getModelBaseLabel(a.base).localeCompare(getModelBaseLabel(b.base));
    case 'size':
      return a.file_size - b.file_size;
    case 'format':
      return String(a.format).localeCompare(String(b.format));
  }
};

export const filterModels = (
  models: ModelConfig[],
  filters: ModelLibraryFilters,
  missingModelKeys: ReadonlySet<string>
): ModelConfig[] => {
  const searchTerm = filters.searchTerm.trim();
  const directionFactor = filters.sortDirection === 'desc' ? -1 : 1;

  return models
    .filter(
      (model) =>
        (!filters.missingOnly || missingModelKeys.has(model.key)) &&
        (filters.typeFilter === null || model.type === filters.typeFilter) &&
        (filters.baseFilter === null || model.base === filters.baseFilter) &&
        matchesSearch(model, searchTerm)
    )
    .sort((a, b) => compareBySortField(a, b, filters.sortField) * directionFactor);
};

/** Group filtered models by category in canonical category order. */
export const groupModelsByType = (models: ModelConfig[]): ModelGroup[] => {
  const groupsByType = new Map<ModelTaxonomyType, ModelConfig[]>();

  for (const model of models) {
    const group = groupsByType.get(model.type);

    if (group) {
      group.push(model);
    } else {
      groupsByType.set(model.type, [model]);
    }
  }

  return [...groupsByType.entries()]
    .sort(([a], [b]) => getModelCategoryRank(a) - getModelCategoryRank(b))
    .map(([type, groupModels]) => ({ label: getModelTypePluralLabel(type), models: groupModels, type }));
};

const matchesPickerSearch = (model: ModelConfig, searchTerm: string): boolean => {
  const terms = searchTerm.trim().toLowerCase().split(/\s+/).filter(Boolean);

  if (terms.length === 0) {
    return true;
  }

  const haystack = `${model.name} ${model.base} ${model.type}`.toLowerCase();

  return terms.every((term) => haystack.includes(term));
};

/** Candidate, search, sort, and grouping rules shared by every model picker instance. */
export const getModelPickerGroups = (models: ModelConfig[], options: ModelPickerOptions): ModelPickerResult => {
  const allowedTypes = new Set(options.modelTypes);
  const candidates = models.filter(
    (model) =>
      allowedTypes.has(model.type) &&
      !options.excludeKeys?.has(model.key) &&
      (options.filter ? options.filter(model) : true)
  );
  // Chips are derived from candidates — before text search and the base filter —
  // so the chip row stays stable while the user types or toggles chips.
  const availableBases = collectBasesForDisplay(candidates);
  const { baseFilter } = options;
  const visibleModels = candidates
    .filter(
      (model) =>
        matchesPickerSearch(model, options.searchTerm) &&
        (!baseFilter || baseFilter.size === 0 || baseFilter.has(String(model.base)))
    )
    .sort(
      (a, b) =>
        getModelCategoryRank(a.type) - getModelCategoryRank(b.type) ||
        getBaseDisplayRank(String(a.base)) - getBaseDisplayRank(String(b.base)) ||
        String(a.base).localeCompare(String(b.base)) ||
        a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
    );

  return { availableBases, candidates, groups: groupModelsForPicker(visibleModels) };
};

const groupModelsForPicker = (models: ModelConfig[]): ModelPickerGroup[] => {
  const groups = new Map<string, ModelPickerGroup>();

  for (const model of models) {
    const base = String(model.base);
    const key = `${model.type}:${base}`;
    const existing = groups.get(key);

    if (existing) {
      existing.models.push(model);
    } else {
      groups.set(key, { base, key, models: [model], type: model.type });
    }
  }

  // Insertion order already follows the sorted model list (category, base, name).
  return [...groups.values()];
};

/** Distinct bases present in a model list, for base filter menus. */
export const collectBases = (models: Pick<ModelConfig, 'base'>[]): string[] =>
  [...new Set(models.map((model) => String(model.base)))].sort();

/** Bases that exist but carry no architecture meaning, sorted after the rest. */
const DEPRIORITIZED_BASES: ReadonlySet<string> = new Set(['any', 'external', 'unknown']);

/**
 * Distinct bases present, ordered for display: known bases follow the
 * registry order from `baseIdentity`, unknown bases come next, and the
 * meaningless `any`/`external`/`unknown` bases are pushed to the very end.
 */
/** Display rank for a base: registry order, unknown bases next, meaningless bases last. */
const getBaseDisplayRank = (base: string): number => {
  if (DEPRIORITIZED_BASES.has(base)) {
    return KNOWN_MODEL_BASES.length + 1;
  }

  const index = KNOWN_MODEL_BASES.indexOf(base as (typeof KNOWN_MODEL_BASES)[number]);

  return index === -1 ? KNOWN_MODEL_BASES.length : index;
};

export const collectBasesForDisplay = (models: Pick<ModelConfig, 'base'>[]): string[] => {
  const bases = [...new Set(models.map((model) => String(model.base)))];

  return bases.sort((a, b) => getBaseDisplayRank(a) - getBaseDisplayRank(b) || a.localeCompare(b));
};

/** Distinct types present in a model list, in canonical order. */
export const collectTypes = (models: Pick<ModelConfig, 'type'>[]): ModelTaxonomyType[] =>
  [...new Set(models.map((model) => model.type))].sort((a, b) => getModelCategoryRank(a) - getModelCategoryRank(b));

/** Rows for a virtualized flat list: group headers interleaved with models. */
export type LibraryRow = { kind: 'header'; group: ModelGroup } | { kind: 'model'; model: ModelConfig };

export const flattenGroupsToRows = (groups: ModelGroup[]): LibraryRow[] =>
  groups.flatMap<LibraryRow>((group) => [
    { group, kind: 'header' },
    ...group.models.map<LibraryRow>((model) => ({ kind: 'model', model })),
  ]);
