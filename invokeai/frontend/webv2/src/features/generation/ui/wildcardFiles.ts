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

import {
  parseWildcardTextFile,
  wildcardsFromNestedRecord,
  wildcardsToNestedRecord,
} from '@features/generation/core/wildcardTransfer';

export type WildcardExportFormat = 'json' | 'yaml';

export const WILDCARD_IMPORT_ACCEPT = '.txt,.yaml,.yml,.json,text/plain,application/json';

/** Directory picking supplies a relative path; a plain pick only has a name. */
const getFilePath = (file: File): string =>
  (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;

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
 * Every wildcard the picked files describe, in the order the files were given.
 *
 * A folder of `.txt` files, a single `.yaml` collection, and a `.json` export of
 * this app all land in the same list — mixing them in one selection is fine, and
 * duplicate names across files are sorted out later by `planWildcardImport`.
 */
export const readWildcardFiles = async (files: readonly File[]): Promise<ParsedWildcard[]> => {
  const parsed: ParsedWildcard[] = [];

  for (const file of files) {
    const path = getFilePath(file);
    const contents = await file.text();

    if (hasExtension(path, '.yaml', '.yml')) {
      const { parse } = await import('yaml');

      try {
        parsed.push(...wildcardsFromNestedRecord(parse(contents)));
      } catch {
        throw new WildcardFileError(file.name);
      }
      continue;
    }

    if (hasExtension(path, '.json')) {
      try {
        parsed.push(...wildcardsFromNestedRecord(JSON.parse(contents)));
      } catch {
        throw new WildcardFileError(file.name);
      }
      continue;
    }

    // Anything else is read a line at a time. A wildcard file has no header to
    // check, and refusing an extensionless one would only be pedantry.
    parsed.push(parseWildcardTextFile(path, contents));
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
  format: WildcardExportFormat
): Promise<void> => {
  const nested = wildcardsToNestedRecord(wildcards);
  const text = format === 'json' ? `${JSON.stringify(nested, null, 2)}\n` : (await import('yaml')).stringify(nested);
  const blob = new Blob([text], { type: format === 'json' ? 'application/json' : 'text/yaml' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');

  anchor.href = url;
  anchor.download = `wildcards.${format}`;
  anchor.click();
  URL.revokeObjectURL(url);
};
