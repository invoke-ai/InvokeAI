import { getModelBaseLabel, getModelCategoryRank, getModelTypePluralLabel } from './taxonomy';
import type { ModelConfig, ModelTaxonomyType } from './types';

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

/** Distinct bases present in a model list, for base filter menus. */
export const collectBases = (models: Pick<ModelConfig, 'base'>[]): string[] =>
  [...new Set(models.map((model) => String(model.base)))].sort();

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
