/**
 * Moving wildcards in and out of the app as files.
 *
 * There are no backend routes for this — wildcards are per-user CRUD, unlike
 * style presets whose transfer is admin-only and server-side — so the whole
 * format layer lives here and the UI issues ordinary creates and updates.
 *
 * Formats follow the `adieyal/dynamicprompts` file conventions, so a catalog can
 * move between this and an a1111 install:
 *
 * - **`.txt`** — one file per wildcard, its path supplying the name and each
 *   line a value. Import only: a folder of them is the shape people already
 *   have, but writing one back per wildcard would mean a download per row.
 * - **`.yaml` / `.json`** — a nested mapping whose arrays are the value lists,
 *   with nesting standing in for `/` in a name. Both directions.
 *
 * Everything here is pure and synchronous, over structures somebody else has
 * already parsed. That keeps the YAML dependency in the UI layer, behind a
 * dynamic import, so it loads only when a file is actually being read or
 * written rather than riding along in the generate widget's chunk.
 */

import { getWildcardNameError, getWildcardValuesError, MAX_WILDCARD_NAME_LENGTH } from './dynamicPrompts';

export interface ParsedWildcard {
  name: string;
  values: string[];
}

/** Mirrors the backend's per-wildcard limits, so a doomed request is never sent. */

/**
 * `animals/dogs.txt` → `animals/dogs`. Directory picking hands back a relative
 * path, which is exactly the nesting the name wants, so it is kept.
 */
export const getWildcardNameFromPath = (path: string): string =>
  path
    .replace(/\.[^./]*$/, '')
    .replace(/^[./]+/, '')
    .trim();

/**
 * One `.txt` file: a value per line, `#` comments and blank lines dropped.
 *
 * `#` is a comment in a prompt too, and a wildcard's values are expanded as
 * prompts, so a value that kept its `#` would silently lose everything after it
 * at generate time anyway.
 */
export const parseWildcardTextFile = (path: string, contents: string): ParsedWildcard => ({
  name: getWildcardNameFromPath(path),
  values: contents
    .split(/\r?\n/)
    .map((line) => line.split('#')[0].trim())
    .filter((line) => line.length > 0),
});

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * A value list, or `null` when this leaf is not one.
 *
 * Scalars are stringified rather than dropped: an unquoted `1970` or `3.5` in a
 * hand-written YAML file is a value the author meant, not a type declaration.
 * Nulls and nested structures are ignored, per the upstream spec.
 */
const toValues = (value: unknown): string[] | null => {
  if (!Array.isArray(value)) {
    return null;
  }

  return value
    .filter((entry): entry is boolean | number | string => ['boolean', 'number', 'string'].includes(typeof entry))
    .map((entry) => String(entry).trim())
    .filter((entry) => entry.length > 0);
};

/**
 * A nested mapping — from YAML or JSON — flattened to `a/b` names.
 *
 * A key that already contains `/` is taken as written, so a flat file and a
 * nested one both round-trip to the same catalog.
 */
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

/** Every prefix of `a/b/c` that could be a wildcard in its own right: `a`, `a/b`. */
const getAncestorNames = (name: string): string[] => {
  const segments = name.split('/');

  return segments.slice(0, -1).map((_, index) => segments.slice(0, index + 1).join('/'));
};

/**
 * The inverse: `animals/dogs` becomes `{ animals: { dogs: [...] } }`.
 *
 * `animals` alongside `animals/dogs` cannot both nest — one key would have to
 * hold a value list and a mapping at once. The *parent* wins its natural key and
 * the child keeps its full name as a flat one, which `wildcardsFromNestedRecord`
 * reads back identically. Deciding this from the name set rather than from
 * whatever the loop has written so far keeps the output independent of the order
 * the wildcards arrive in.
 *
 * Every level is a null-prototype object, and levels are created through an own
 * key check rather than `??=`. A name segment is user text, and `constructor`,
 * `toString` and `valueOf` are all valid wildcard names: on an ordinary `{}` the
 * inherited value is truthy, so `??=` would decline to assign and the walk would
 * step onto `Object.prototype` and write the leaf there. `Object.keys` becoming
 * an array of colours takes the whole tab down, and the wildcard vanishes from
 * the export either way.
 */
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
  /** The wildcard this would overwrite, or `null` when the name is new. */
  conflictId: string | null;
  /** Why this entry cannot be imported at all, or `null`. */
  rejection: WildcardImportRejection | null;
}

/**
 * What a file would do to the catalog, before anything is sent.
 *
 * Nothing is dropped quietly. A name the backend would refuse, a list that is
 * empty or too long, and a name repeated within the same import all come back
 * as entries carrying their reason, so the dialog can say what it is not going
 * to do rather than appearing to have imported everything.
 */
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
    // An empty name means the file was named something unusable, which is the
    // same problem as an invalid one from the reader's side.
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

    // Only entries that will actually be written claim the name. Recording
    // rejected ones too meant an empty `colours.txt` earlier in the selection
    // made a populated `colours.yaml` later in it read as a repeat, and neither
    // was imported.
    if (rejection === null) {
      seen.add(wildcard.name.trim());
    }

    return {
      conflictId: existingByName.get(wildcard.name.trim()) ?? null,
      // The trimmed form throughout: validation trims before deciding, so
      // storing the untrimmed one would send the server something the client
      // never checked, and look it up under a name it was not filed under.
      name: wildcard.name.trim(),
      rejection,
      values: wildcard.values,
    };
  });
};

/**
 * `colours` → `colours-2`, then `-3`, and so on.
 *
 * The base is trimmed rather than the suffix dropped when the two together
 * would exceed the backend's length limit — a name one character over the line
 * is not a reason to refuse the import.
 */
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
  /** Present for a replacement; absent when this creates a new wildcard. */
  id?: string;
}

/**
 * The writes a planned import resolves to, in order.
 *
 * `taken` grows as it goes, so importing two clashing names as "keep both" gives
 * `colours-2` and `colours-3` rather than the same suffix twice.
 *
 * It is seeded with the names the import itself is about to claim, not just the
 * ones already in the catalog. A file holding `colours` (clashing) and a new
 * `colours-2` otherwise had the first resolve to `colours-2` as well, and the
 * two creates raced each other to the same name.
 */
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
