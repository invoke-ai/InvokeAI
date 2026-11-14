import type { ModelConfig, ModelTaxonomyType } from './types';

import { getModelBaseLabel, KNOWN_MODEL_BASES } from './baseIdentity';
import { getModelCategoryRank, getModelTypePluralLabel } from './taxonomy';

export type ModelSortField = 'default' | 'name' | 'base' | 'size' | 'format' | 'type' | 'path';

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

/** A model listed in a type group; `viaType` is the role type when the group is a role type of a model listed there as well as under its primary type. */
export interface ModelGroupEntry {
  model: ModelConfig;
  viaType: ModelTaxonomyType | null;
}

export interface ModelGroup {
  type: ModelTaxonomyType;
  label: string;
  entries: ModelGroupEntry[];
}

type WithRoles = { type: ModelTaxonomyType; roles?: readonly { type: ModelTaxonomyType }[] };

/** A model serves a type as its primary type or through a role. */
export const modelServesType = (model: WithRoles, type: ModelTaxonomyType): boolean =>
  model.type === type || (model.roles ?? []).some((role) => role.type === type);

export interface ModelPickerOptions {
  /** Multi-select base filter; empty/undefined = all bases. */
  baseFilter?: ReadonlySet<string>;
  /** Hide specific models, e.g. the current model or already-linked choices. */
  excludeKeys?: ReadonlySet<string>;
  /** Extra predicate supplied by the owning form, e.g. base compatibility. */
  filter?: (model: ModelConfig) => boolean;
  modelTypes: readonly ModelTaxonomyType[];
  /** Models linked to the current selection, floated to the top of their group. */
  relatedKeys?: ReadonlySet<string>;
}

/** Group by base; multi-type rows carry badges rather than repeated type section headings. */
export interface ModelPickerGroup {
  key: string;
  base: string;
  models: ModelConfig[];
}

export interface ModelPickerResult {
  /** Distinct bases among candidates (pre-search, pre-base-filter), for chips. */
  availableBases: string[];
  /** Models available before the base filter (text search is the Picker's own concern), used for empty-state copy. */
  candidates: ModelConfig[];
  groups: ModelPickerGroup[];
}

const matchesSearchTerms = (model: ModelConfig, terms: readonly string[]): boolean => {
  if (terms.length === 0) {
    return true;
  }

  const haystack =
    `${model.name} ${model.description ?? ''} ${model.base} ${model.type} ${(model.roles ?? []).map((role) => role.type).join(' ')} ${model.format} ${(model.trigger_phrases ?? []).join(' ')}`.toLowerCase();

  return terms.every((term) => haystack.includes(term));
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
    case 'type':
      return (
        getModelCategoryRank(a.type) - getModelCategoryRank(b.type) ||
        a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
      );
    case 'path':
      return a.path.localeCompare(b.path);
  }
};

export const filterModels = (
  models: ModelConfig[],
  filters: ModelLibraryFilters,
  missingModelKeys: ReadonlySet<string>
): ModelConfig[] => {
  // Split once per pass, not once per model.
  const terms = filters.searchTerm.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const directionFactor = filters.sortDirection === 'desc' ? -1 : 1;

  return models
    .filter(
      (model) =>
        (!filters.missingOnly || missingModelKeys.has(model.key)) &&
        (filters.typeFilter === null || modelServesType(model, filters.typeFilter)) &&
        (filters.baseFilter === null || model.base === filters.baseFilter) &&
        matchesSearchTerms(model, terms)
    )
    .sort((a, b) => compareBySortField(a, b, filters.sortField) * directionFactor);
};

/**
 * Group filtered models by category in canonical category order. A model also sits in each of its role types'
 * groups; under a type filter it sits in the filtered type's group only, as a plain entry.
 */
export const groupModelsByType = (models: ModelConfig[], typeFilter: ModelTaxonomyType | null = null): ModelGroup[] => {
  const groupsByType = new Map<ModelTaxonomyType, ModelGroupEntry[]>();
  const add = (type: ModelTaxonomyType, entry: ModelGroupEntry) => {
    const group = groupsByType.get(type);

    if (group) {
      group.push(entry);
    } else {
      groupsByType.set(type, [entry]);
    }
  };

  for (const model of models) {
    if (typeFilter !== null) {
      // Listed once, so it is the model's only row: no section-qualified identity needed.
      add(typeFilter, { model, viaType: null });
      continue;
    }

    add(model.type, { model, viaType: null });

    for (const roleType of new Set((model.roles ?? []).map((role) => role.type))) {
      if (roleType !== model.type) {
        add(roleType, { model, viaType: roleType });
      }
    }
  }

  return [...groupsByType.entries()]
    .sort(([a], [b]) => getModelCategoryRank(a) - getModelCategoryRank(b))
    .map(([type, entries]) => ({ entries, label: getModelTypePluralLabel(type), type }));
};

/** The type a candidate qualifies as in a picker: its primary type if asked for, else its first asked-for role type. */
export const getQualifyingType = (model: WithRoles, allowedTypes: readonly string[]): ModelTaxonomyType =>
  allowedTypes.includes(model.type)
    ? model.type
    : ((model.roles ?? []).find((role) => allowedTypes.includes(role.type))?.type ?? model.type);

/** Candidate, sort, and grouping rules shared by every model picker instance. */
const isModelPickerCandidate = (
  model: ModelConfig,
  allowedTypes: ReadonlySet<string>,
  options: Pick<ModelPickerOptions, 'excludeKeys' | 'filter'>
): boolean =>
  (allowedTypes.has(model.type) || (model.roles ?? []).some((role) => allowedTypes.has(role.type))) &&
  !options.excludeKeys?.has(model.key) &&
  (options.filter ? options.filter(model) : true);

/** Whether the picker would list anything at all — the base filter aside, which only narrows a non-empty list. */
export const hasModelPickerCandidates = (
  models: readonly ModelConfig[],
  options: Pick<ModelPickerOptions, 'excludeKeys' | 'filter' | 'modelTypes'>
): boolean => {
  const allowedTypes = new Set(options.modelTypes);

  return models.some((model) => isModelPickerCandidate(model, allowedTypes, options));
};

export const getModelPickerGroups = (models: ModelConfig[], options: ModelPickerOptions): ModelPickerResult => {
  const allowedTypes = new Set(options.modelTypes);
  const candidates = models.filter((model) => isModelPickerCandidate(model, allowedTypes, options));
  // Chips are derived from candidates — before the base filter — so the chip
  // row stays stable while the user toggles chips.
  const availableBases = collectBasesForDisplay(candidates);
  // A remembered base that this picker does not offer (uninstalled, or a
  // sibling picker's choice) must not filter everything out behind a hidden chip.
  const baseFilter = new Set(availableBases.filter((base) => options.baseFilter?.has(base)));
  const { relatedKeys } = options;
  const qualifyingTypes = new Map(candidates.map((model) => [model, getQualifyingType(model, options.modelTypes)]));
  const isNative = (model: ModelConfig) => qualifyingTypes.get(model) === model.type;
  const visibleModels = candidates
    .filter((model) => baseFilter.size === 0 || baseFilter.has(String(model.base)))
    .sort(
      (a, b) =>
        getBaseDisplayRank(String(a.base)) - getBaseDisplayRank(String(b.base)) ||
        String(a.base).localeCompare(String(b.base)) ||
        // Models of the asked type lead; those that only serve as it follow.
        Number(isNative(b)) - Number(isNative(a)) ||
        Number(relatedKeys?.has(b.key) ?? false) - Number(relatedKeys?.has(a.key) ?? false) ||
        getModelCategoryRank(qualifyingTypes.get(a)!) - getModelCategoryRank(qualifyingTypes.get(b)!) ||
        a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
    );

  return { availableBases, candidates, groups: groupModelsForPicker(visibleModels) };
};

const groupModelsForPicker = (models: ModelConfig[]): ModelPickerGroup[] => {
  const groups = new Map<string, ModelPickerGroup>();

  for (const model of models) {
    const base = String(model.base);
    const existing = groups.get(base);

    if (existing) {
      existing.models.push(model);
    } else {
      groups.set(base, { base, key: base, models: [model] });
    }
  }

  // Insertion order already follows the sorted model list (base, related, name).
  return [...groups.values()];
};

/** Distinct bases present in a model list, for base filter menus. */
export const collectBases = (models: Pick<ModelConfig, 'base'>[]): string[] =>
  [...new Set(models.map((model) => String(model.base)))].sort();

/** Bases that exist but carry no architecture meaning, sorted after the rest. */
const DEPRIORITIZED_BASES: ReadonlySet<string> = new Set(['any', 'external', 'unknown']);

/** Order known bases by registry, then unfamiliar bases, with any/external/unknown last. */
const KNOWN_BASE_RANKS = new Map<string, number>(KNOWN_MODEL_BASES.map((base, index) => [base, index]));

/** Display rank for a base: registry order, unknown bases next, meaningless bases last. */
const getBaseDisplayRank = (base: string): number => {
  if (DEPRIORITIZED_BASES.has(base)) {
    return KNOWN_MODEL_BASES.length + 1;
  }

  return KNOWN_BASE_RANKS.get(base) ?? KNOWN_MODEL_BASES.length;
};

export const collectBasesForDisplay = (models: Pick<ModelConfig, 'base'>[]): string[] => {
  const bases = [...new Set(models.map((model) => String(model.base)))];

  return bases.sort((a, b) => getBaseDisplayRank(a) - getBaseDisplayRank(b) || a.localeCompare(b));
};

/** Distinct types present in a model list, in canonical order. */
export const collectTypes = (models: WithRoles[]): ModelTaxonomyType[] =>
  [...new Set(models.flatMap((model) => [model.type, ...(model.roles ?? []).map((role) => role.type)]))].sort(
    (a, b) => getModelCategoryRank(a) - getModelCategoryRank(b)
  );
