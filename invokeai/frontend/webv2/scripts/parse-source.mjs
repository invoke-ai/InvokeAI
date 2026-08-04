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

let fileSystem;
let api;
let snapshot;
let nextId = 0;
const virtualPaths = new Map();
const stagedText = new Map();
const created = [];
const changed = [];

const ensureSession = () => {
  if (!api) {
    fileSystem = createVirtualFileSystem({ [CONFIG_PATH]: CONFIG, [SEED_PATH]: 'export {};' });
    api = new API({ cwd: PARSE_ROOT, fs: fileSystem });
    snapshot = api.updateSnapshot({ openProjects: [CONFIG_PATH] });
  }
  return api;
};

/**
 * Logical paths are analyzer-facing labels rather than real files, so each one
 * gets an opaque virtual path. The extension carries the only parse-visible
 * decision: `.tsx` parses `<T>` as JSX, `.ts` parses it as a type assertion.
 */
const virtualPathFor = (fileName, jsx) => {
  const key = `${jsx ? 'x' : ''}:${fileName}`;
  let virtualPath = virtualPaths.get(key);
  if (!virtualPath) {
    virtualPath = `${PARSE_ROOT}/${nextId++}.${jsx ? 'tsx' : 'ts'}`;
    virtualPaths.set(key, virtualPath);
  }
  return virtualPath;
};

const stage = (fileName, text, jsx) => {
  ensureSession();
  const virtualPath = virtualPathFor(fileName, jsx);
  if (stagedText.get(virtualPath) === text) {
    return virtualPath;
  }
  (stagedText.has(virtualPath) ? changed : created).push(virtualPath);
  stagedText.set(virtualPath, text);
  fileSystem.writeFile(virtualPath, text);
  return virtualPath;
};

const flush = () => {
  if (created.length === 0 && changed.length === 0) {
    return;
  }
  snapshot = api.updateSnapshot({
    fileChanges: { changed: [...changed], created: [...created] },
    openProjects: [CONFIG_PATH],
  });
  created.length = 0;
  changed.length = 0;
};

const read = (virtualPath, fileName) => {
  const sourceFile = snapshot.getProjects()[0]?.program.getSourceFile(virtualPath);
  if (!sourceFile) {
    throw new Error(`Failed to parse ${fileName} through the TypeScript API.`);
  }
  return sourceFile;
};

const resolveJsx = (fileName, jsx) => jsx ?? fileName.endsWith('x');

/**
 * Stage many sources and load them with one snapshot update. Use this before a
 * whole-tree sweep; parsing each file on its own costs a round trip apiece.
 */
export const primeSources = (entries, { jsx } = {}) => {
  for (const [fileName, text] of entries) {
    stage(fileName, text, resolveJsx(fileName, jsx));
  }
  flush();
};

export const parseSource = (fileName, text, { jsx } = {}) => {
  const virtualPath = stage(fileName, text, resolveJsx(fileName, jsx));
  flush();
  return read(virtualPath, fileName);
};
