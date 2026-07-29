import type { ParsedWildcard } from '@features/generation/core/wildcardTransfer';
import type { AccountScope } from '@platform/state/accountLifecycle';

import {
  parseWildcardTextFile,
  wildcardsFromNestedRecord,
  wildcardsToNestedRecord,
} from '@features/generation/core/wildcardTransfer';
import { downloadText } from '@platform/browser/downloadBlob';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';

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

export class WildcardFileError extends Error {
  constructor(readonly fileName: string) {
    super(`Could not read ${fileName}`);
    this.name = 'WildcardFileError';
  }
}

const MAX_IMPORT_FILE_BYTES = 32 * 1024 * 1024;
const MAX_IMPORT_WILDCARDS = 20_000;

const isTextWildcardFile = (path: string): boolean => {
  const name = path.slice(path.lastIndexOf('/') + 1);
  const dot = name.lastIndexOf('.');

  return dot < 0 || name.slice(dot).toLowerCase() === '.txt';
};

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

    if (!isTextWildcardFile(path)) {
      throw new WildcardFileError(file.name);
    }

    collect([parseWildcardTextFile(path, contents)], file.name);
  }

  return parsed;
};

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
