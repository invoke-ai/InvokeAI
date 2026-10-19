import {
  getWildcardNameError,
  getWildcardValuesError,
  MAX_WILDCARD_NAME_LENGTH,
  normalizeWildcardValues,
} from './dynamicPrompts';

export interface ParsedWildcard {
  name: string;
  values: string[];
}

export const getWildcardNameFromPath = (path: string): string =>
  path
    .replace(/\.[^./]*$/, '')
    .replace(/^[./]+/, '')
    .trim();

export const parseWildcardTextFile = (path: string, contents: string): ParsedWildcard => ({
  name: getWildcardNameFromPath(path),
  values: normalizeWildcardValues(contents.split(/\r?\n/)),
});

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const toValues = (value: unknown): string[] | null => {
  if (!Array.isArray(value)) {
    return null;
  }

  return normalizeWildcardValues(
    value
      .filter((entry): entry is boolean | number | string => ['boolean', 'number', 'string'].includes(typeof entry))
      .map((entry) => String(entry))
  );
};

export const wildcardsFromNestedRecord = (value: unknown): ParsedWildcard[] => {
  const wildcards: ParsedWildcard[] = [];

  const walk = (node: unknown, prefix: string): void => {
    if (!isPlainObject(node)) {
      return;
    }

    for (const [key, child] of Object.entries(node)) {
      const name = prefix ? `${prefix}/${key}` : key;
      const values = toValues(child);

      if (values) {
        wildcards.push({ name: name.trim(), values });
      } else {
        walk(child, name);
      }
    }
  };

  walk(value, '');

  return wildcards;
};

const getAncestorNames = (name: string): string[] => {
  const segments = name.split('/');

  return segments.slice(0, -1).map((_, index) => segments.slice(0, index + 1).join('/'));
};

// Every path segment is user text, so levels must not inherit prototype keys.
export const wildcardsToNestedRecord = (wildcards: readonly ParsedWildcard[]): Record<string, unknown> => {
  const root: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
  const names = new Set(wildcards.map((wildcard) => wildcard.name));

  for (const wildcard of wildcards) {
    if (getAncestorNames(wildcard.name).some((ancestor) => names.has(ancestor))) {
      root[wildcard.name] = wildcard.values;
      continue;
    }

    const segments = wildcard.name.split('/');
    const leaf = segments.pop();

    if (leaf === undefined) {
      continue;
    }

    let node = root;

    for (const segment of segments) {
      if (!Object.hasOwn(node, segment)) {
        node[segment] = Object.create(null) as Record<string, unknown>;
      }

      node = node[segment] as Record<string, unknown>;
    }

    node[leaf] = wildcard.values;
  }

  return root;
};

export type WildcardImportRejection =
  | 'duplicate'
  | 'invalid'
  | 'noValues'
  | 'tooLong'
  | 'tooManyValues'
  | 'valueTooLong';

export type WildcardImportResolution = 'keepBoth' | 'replace' | 'skip';

export interface WildcardImportEntry {
  name: string;
  values: string[];
  conflictId: string | null;
  rejection: WildcardImportRejection | null;
}

export const planWildcardImport = (
  parsed: readonly ParsedWildcard[],
  existing: readonly { id: string; name: string }[]
): WildcardImportEntry[] => {
  const existingByName = new Map(existing.map((wildcard) => [wildcard.name, wildcard.id]));
  const seen = new Set<string>();

  const getRejection = (wildcard: ParsedWildcard): WildcardImportRejection | null => {
    const nameError = getWildcardNameError(wildcard.name);

    if (nameError === 'tooLong') {
      return 'tooLong';
    }
    if (nameError !== null) {
      return 'invalid';
    }
    if (seen.has(wildcard.name.trim())) {
      return 'duplicate';
    }
    if (wildcard.values.length === 0) {
      return 'noValues';
    }
    return getWildcardValuesError(wildcard.values);
  };

  return parsed.map((wildcard) => {
    const rejection = getRejection(wildcard);

    if (rejection === null) {
      seen.add(wildcard.name.trim());
    }

    return {
      conflictId: existingByName.get(wildcard.name.trim()) ?? null,
      name: wildcard.name.trim(),
      rejection,
      values: wildcard.values,
    };
  });
};

export const getAvailableWildcardName = (name: string, taken: ReadonlySet<string>): string => {
  for (let suffix = 2; ; suffix++) {
    const marker = `-${suffix}`;
    const base = name.slice(0, MAX_WILDCARD_NAME_LENGTH - marker.length).replace(/[^A-Za-z0-9]+$/, '');
    const candidate = `${base}${marker}`;

    if (!taken.has(candidate)) {
      return candidate;
    }
  }
};

export interface WildcardImportAction {
  name: string;
  values: string[];
  id?: string;
}

export const getWildcardImportActions = (
  entries: readonly WildcardImportEntry[],
  resolutions: Readonly<Record<string, WildcardImportResolution>>,
  existingNames: ReadonlySet<string>
): WildcardImportAction[] => {
  const taken = new Set(existingNames);
  const actions: WildcardImportAction[] = [];

  for (const entry of entries) {
    if (!entry.rejection) {
      taken.add(entry.name);
    }
  }

  for (const entry of entries) {
    if (entry.rejection) {
      continue;
    }

    if (entry.conflictId === null) {
      actions.push({ name: entry.name, values: entry.values });
      continue;
    }

    const resolution = resolutions[entry.name] ?? 'skip';

    if (resolution === 'skip') {
      continue;
    }

    if (resolution === 'replace') {
      actions.push({ id: entry.conflictId, name: entry.name, values: entry.values });
      continue;
    }

    const name = getAvailableWildcardName(entry.name, taken);

    taken.add(name);
    actions.push({ name, values: entry.values });
  }

  return actions;
};
