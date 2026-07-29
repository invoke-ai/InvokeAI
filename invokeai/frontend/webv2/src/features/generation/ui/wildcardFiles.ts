/**
 * The file half of wildcard import/export: reading what the user picked, and
 * handing back a download.
 *
 * The format logic itself lives in `core/wildcardTransfer` — this module only
 * bridges it to `File`, `Blob` and YAML text. YAML is loaded through a dynamic
 * `import('yaml')` so the parser reaches the browser only when somebody actually
 * imports or exports, rather than riding along in the generate widget's chunk.
 */

import type { ParsedWildcard } from '@features/generation/core/wildcardTransfer';
import type { AccountScope } from '@platform/state/accountLifecycle';

import {
  parseWildcardTextFile,
  wildcardsFromNestedRecord,
  wildcardsToNestedRecord,
} from '@features/generation/core/wildcardTransfer';
import { downloadText } from '@platform/browser/downloadBlob';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';

/**
 * Every collection format, stated once.
 *
 * Adding one used to mean editing five places that had no idea about each
 * other: a union type, two ternaries here, the export menu's array, and an i18n
 * key built by string concatenation — which silently fell back to "Json" for
 * anything it did not recognise rather than failing to compile.
 */
export const WILDCARD_COLLECTION_FORMATS = [
  {
    extensions: ['.yaml', '.yml'],
    id: 'yaml',
    labelKey: 'widgets.generate.dynamicPrompts.exportAsYaml',
    mimeType: 'text/yaml',
    parse: async (text: string): Promise<unknown> => (await import('yaml')).parse(text),
    stringify: async (value: unknown): Promise<string> => (await import('yaml')).stringify(value),
  },
  {
    extensions: ['.json'],
    id: 'json',
    labelKey: 'widgets.generate.dynamicPrompts.exportAsJson',
    mimeType: 'application/json',
    parse: (text: string): Promise<unknown> => Promise.resolve(JSON.parse(text) as unknown),
    stringify: (value: unknown): Promise<string> => Promise.resolve(`${JSON.stringify(value, null, 2)}\n`),
  },
] as const;

export type WildcardExportFormat = (typeof WILDCARD_COLLECTION_FORMATS)[number]['id'];

export const WILDCARD_IMPORT_ACCEPT = [
  '.txt',
  ...WILDCARD_COLLECTION_FORMATS.flatMap((format) => format.extensions),
  'text/plain',
  'application/json',
].join(',');

export type WildcardFileSource = 'files' | 'folder';

/** Folder picks shed the selected root while direct picks retain their file names. */
const getFilePath = (file: File, source: WildcardFileSource): string => {
  if (source === 'files') {
    return file.name;
  }

  const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
  const firstSeparator = relativePath.indexOf('/');

  return firstSeparator < 0 ? relativePath : relativePath.slice(firstSeparator + 1);
};

const hasExtension = (path: string, ...extensions: string[]): boolean =>
  extensions.some((extension) => path.toLowerCase().endsWith(extension));

/**
 * Thrown when a file cannot be read as any supported format. Carries the file
 * name so the notice can say which one, since imports are usually plural.
 */
export class WildcardFileError extends Error {
  constructor(readonly fileName: string) {
    super(`Could not read ${fileName}`);
    this.name = 'WildcardFileError';
  }
}

/**
 * Bounds on what a selection may contain, so a mistaken pick fails by saying so.
 *
 * Without the count, `parsed.push(...entries)` on a large collection overflowed
 * the argument stack and surfaced as `RangeError` — caught as a parse failure
 * and reported as "could not read", about a file that had parsed perfectly. The
 * threshold is engine-dependent, so it tripped in Safari well before Chrome.
 */
const MAX_IMPORT_FILE_BYTES = 32 * 1024 * 1024;
const MAX_IMPORT_WILDCARDS = 20_000;

/**
 * Every wildcard the picked files describe, in the order the files were given.
 *
 * A folder of `.txt` files, a single `.yaml` collection, and a `.json` export of
 * this app all land in the same list — mixing them in one selection is fine, and
 * duplicate names across files are sorted out later by `planWildcardImport`.
 */
/**
 * `a.txt` and a bare `LICENSE` yes; `README.md`, `preview.png` and `.DS_Store`
 * no. An extensionless file is fair game — a wildcard file has no header to
 * check — but a name that is nothing *but* an extension is a dotfile, and every
 * folder pick brings a few of those along.
 */
const isTextWildcardFile = (path: string): boolean => {
  const name = path.slice(path.lastIndexOf('/') + 1);
  const dot = name.lastIndexOf('.');

  return dot < 0 || name.slice(dot).toLowerCase() === '.txt';
};

/**
 * Whether this file is worth handing to `readWildcardFiles` at all.
 *
 * For a file the user picked by name, being unreadable is worth saying out loud,
 * so `readWildcardFiles` throws. A directory pick is the opposite: it hands over
 * everything in the folder, readmes and `.DS_Store` included, and refusing the
 * whole import over one of them would make folder picking useless.
 */
export const isSupportedWildcardFile = (file: File): boolean => {
  const path = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;

  return (
    WILDCARD_COLLECTION_FORMATS.some((format) => hasExtension(path, ...format.extensions)) || isTextWildcardFile(path)
  );
};

export const readWildcardFiles = async (
  files: readonly File[],
  source: WildcardFileSource,
  owner: AccountScope
): Promise<ParsedWildcard[]> => {
  const parsed: ParsedWildcard[] = [];

  const collect = (wildcards: readonly ParsedWildcard[], fileName: string): void => {
    // Appended one at a time rather than spread: a spread of a large collection
    // is a call with that many arguments, which overflows the stack.
    for (const wildcard of wildcards) {
      parsed.push(wildcard);
    }

    if (parsed.length > MAX_IMPORT_WILDCARDS) {
      throw new WildcardFileError(fileName);
    }
  };

  for (const file of files) {
    const path = getFilePath(file, source);

    if (file.size > MAX_IMPORT_FILE_BYTES) {
      throw new WildcardFileError(file.name);
    }

    const contents = await file.text();
    assertAccountScopeCurrent(owner);

    const collectionFormat = WILDCARD_COLLECTION_FORMATS.find((format) => hasExtension(path, ...format.extensions));

    if (collectionFormat) {
      let collection: unknown;

      try {
        collection = await collectionFormat.parse(contents);
      } catch {
        throw new WildcardFileError(file.name);
      }

      assertAccountScopeCurrent(owner);

      try {
        collect(wildcardsFromNestedRecord(collection), file.name);
      } catch {
        throw new WildcardFileError(file.name);
      }
      continue;
    }

    // Text is read a line at a time, and an extensionless file is fair game — a
    // wildcard file has no header to check. Anything else is not: `accept` is
    // only a hint on a file input, and a README picked up alongside a wildcard
    // folder used to import as a wildcard named `README`.
    if (!isTextWildcardFile(path)) {
      throw new WildcardFileError(file.name);
    }

    collect([parseWildcardTextFile(path, contents)], file.name);
  }

  return parsed;
};

/**
 * Downloads the whole catalog.
 *
 * There is no `.txt` counterpart on purpose: the convention is one file per
 * wildcard, so a text export of forty wildcards would be forty downloads. Both
 * formats here keep the nesting, and a1111's extension reads YAML collections,
 * so the round trip that matters is covered.
 */
export const downloadWildcards = async (
  wildcards: readonly ParsedWildcard[],
  formatId: WildcardExportFormat,
  owner: AccountScope
): Promise<void> => {
  const format = WILDCARD_COLLECTION_FORMATS.find((candidate) => candidate.id === formatId);

  if (!format) {
    throw new Error(`Unknown wildcard export format: ${formatId}`);
  }

  const contents = await format.stringify(wildcardsToNestedRecord(wildcards));

  assertAccountScopeCurrent(owner);
  downloadText(contents, `wildcards.${format.id}`, format.mimeType);
};
