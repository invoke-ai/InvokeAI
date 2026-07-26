import { relative, resolve, sep } from 'node:path';

const SCHEMA_VERSION = 1;
const VIRTUAL_MODULE_PREFIX = '\0';

const toPosixPath = (value) => value.split(sep).join('/');

const getPackageOwner = (moduleId) => {
  const marker = '/node_modules/';
  const markerIndex = moduleId.lastIndexOf(marker);

  if (markerIndex === -1) {
    return null;
  }

  const packagePath = moduleId.slice(markerIndex + marker.length);
  const segments = packagePath.split('/');
  const packageName = segments[0]?.startsWith('@') ? segments.slice(0, 2).join('/') : segments[0];

  return packageName ? `package:${packageName}` : null;
};

/**
 * Converts a bundler module id into a stable ownership id.
 *
 * First-party modules retain their repo-relative source path. Dependency
 * internals collapse to the package that owns them, so a lockfile refresh does
 * not rewrite the graph merely because pnpm's content-addressed path changed.
 * Virtual/runtime modules are intentionally excluded: they have no source
 * owner and their emitted bytes are still covered by the outcome budgets.
 */
export const getModuleSourceOwner = (moduleId, projectRoot) => {
  if (typeof moduleId !== 'string' || moduleId.length === 0 || moduleId.startsWith(VIRTUAL_MODULE_PREFIX)) {
    return null;
  }

  const pathWithoutQuery = moduleId.split('?')[0];
  const normalizedPath = toPosixPath(pathWithoutQuery);
  const packageOwner = getPackageOwner(normalizedPath);

  if (packageOwner) {
    return packageOwner;
  }

  const normalizedRoot = resolve(projectRoot);
  const relativePath = toPosixPath(relative(normalizedRoot, pathWithoutQuery));

  if (relativePath === '' || relativePath.startsWith('../') || relativePath === '..') {
    return null;
  }

  return `source:${relativePath}`;
};

export const createChunkSourceManifest = (bundle, projectRoot) => {
  const chunks = {};

  for (const output of Object.values(bundle)) {
    if (output.type !== 'chunk') {
      continue;
    }

    const sourceOwners = new Set(
      output.moduleIds
        .map((moduleId) => getModuleSourceOwner(moduleId, projectRoot))
        .filter((sourceOwner) => sourceOwner !== null)
    );

    chunks[output.fileName] = {
      facadeSource: getModuleSourceOwner(output.facadeModuleId, projectRoot),
      sourceOwners: [...sourceOwners].sort(),
    };
  }

  return {
    chunks: Object.fromEntries(Object.entries(chunks).sort(([left], [right]) => left.localeCompare(right))),
    schemaVersion: SCHEMA_VERSION,
  };
};

export const chunkSourceManifest = ({ projectRoot }) => ({
  name: 'invokeai-chunk-source-manifest',
  generateBundle(_outputOptions, bundle) {
    const manifest = createChunkSourceManifest(bundle, projectRoot);

    this.emitFile({
      fileName: '.vite/chunk-sources.json',
      source: `${JSON.stringify(manifest, null, 2)}\n`,
      type: 'asset',
    });
  },
});
