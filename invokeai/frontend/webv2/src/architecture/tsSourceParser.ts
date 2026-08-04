import type { SourceFile } from 'typescript/unstable/ast';

import { createVirtualFileSystem } from 'typescript/unstable/fs';
import { API } from 'typescript/unstable/sync';

/**
 * TypeScript 7 removed the in-process `ts.createSourceFile` parser. The
 * replacement surface only hands out ASTs for files that belong to a program,
 * so the architecture analyzers parse through a virtual filesystem backed by a
 * single long-lived tsgo session instead of parsing detached strings.
 *
 * Files are staged into the virtual project and flushed in one `updateSnapshot`
 * call, which keeps a whole-tree sweep to a single round trip.
 */
const PARSE_ROOT = '/architecture-sources';
const CONFIG_PATH = `${PARSE_ROOT}/tsconfig.json`;
const SEED_PATH = `${PARSE_ROOT}/seed.tsx`;

// `noLib` keeps lib.d.ts out of the program: these analyzers only ever walk
// syntax, so skipping the default library removes the bulk of the load cost.
const CONFIG = JSON.stringify({
  compilerOptions: { allowJs: true, jsx: 'preserve', module: 'esnext', noLib: true, target: 'esnext' },
  include: ['**/*'],
});

interface ParseOptions {
  /** Defaults to the logical file name's extension. */
  jsx?: boolean;
}

let fileSystem: ReturnType<typeof createVirtualFileSystem> | undefined;
let api: API | undefined;
let snapshot: ReturnType<API['updateSnapshot']> | undefined;
let nextId = 0;
const virtualPaths = new Map<string, string>();
const stagedText = new Map<string, string>();
const created: string[] = [];
const changed: string[] = [];

const ensureSession = (): void => {
  if (!api) {
    fileSystem = createVirtualFileSystem({ [CONFIG_PATH]: CONFIG, [SEED_PATH]: 'export {};' });
    api = new API({ cwd: PARSE_ROOT, fs: fileSystem });
    snapshot = api.updateSnapshot({ openProjects: [CONFIG_PATH] });
  }
};

/**
 * Logical paths are analyzer-facing labels rather than real files, so each one
 * gets an opaque virtual path. The extension carries the only parse-visible
 * decision: `.tsx` parses `<T>` as JSX, `.ts` parses it as a type assertion.
 */
const virtualPathFor = (fileName: string, jsx: boolean): string => {
  const key = `${jsx ? 'x' : ''}:${fileName}`;
  let virtualPath = virtualPaths.get(key);
  if (!virtualPath) {
    virtualPath = `${PARSE_ROOT}/${nextId++}.${jsx ? 'tsx' : 'ts'}`;
    virtualPaths.set(key, virtualPath);
  }
  return virtualPath;
};

const stage = (fileName: string, text: string, jsx: boolean): string => {
  ensureSession();
  const virtualPath = virtualPathFor(fileName, jsx);
  if (stagedText.get(virtualPath) === text) {
    return virtualPath;
  }
  (stagedText.has(virtualPath) ? changed : created).push(virtualPath);
  stagedText.set(virtualPath, text);
  fileSystem?.writeFile?.(virtualPath, text);
  return virtualPath;
};

const flush = (): void => {
  if (created.length === 0 && changed.length === 0) {
    return;
  }
  snapshot = api?.updateSnapshot({
    fileChanges: { changed: [...changed], created: [...created] },
    openProjects: [CONFIG_PATH],
  });
  created.length = 0;
  changed.length = 0;
};

const read = (virtualPath: string, fileName: string): SourceFile => {
  const sourceFile = snapshot?.getProjects()[0]?.program.getSourceFile(virtualPath);
  if (!sourceFile) {
    throw new Error(`Failed to parse ${fileName} through the TypeScript API.`);
  }
  return sourceFile;
};

const resolveJsx = (fileName: string, jsx: boolean | undefined): boolean => jsx ?? fileName.endsWith('x');

/**
 * Stage many sources and load them with one snapshot update. Use this before a
 * whole-tree sweep; parsing each file on its own costs a round trip apiece.
 */
export const primeSources = (entries: Iterable<readonly [string, string]>, { jsx }: ParseOptions = {}): void => {
  for (const [fileName, text] of entries) {
    stage(fileName, text, resolveJsx(fileName, jsx));
  }
  flush();
};

export const parseSource = (fileName: string, text: string, { jsx }: ParseOptions = {}): SourceFile => {
  const virtualPath = stage(fileName, text, resolveJsx(fileName, jsx));
  flush();
  return read(virtualPath, fileName);
};
