import {
  downloadWildcards,
  isSupportedWildcardFile,
  readWildcardFiles,
  WILDCARD_COLLECTION_FORMATS,
  WildcardFileError,
} from '@features/generation/ui/wildcardFiles';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, describe, expect, it, vi } from 'vitest';

const downloadText = vi.hoisted(() => vi.fn());

vi.mock('@platform/browser/downloadBlob', () => ({ downloadText }));

const file = (name: string, contents: string) => new File([contents], name, { type: 'text/plain' });
const read = (files: readonly File[], source: 'files' | 'folder') =>
  readWildcardFiles(files, source, accountLifecycle.capture());

const inFolder = (path: string, contents: string): File => {
  const picked = file(path.slice(path.lastIndexOf('/') + 1), contents);

  Object.defineProperty(picked, 'webkitRelativePath', { value: path });

  return picked;
};

afterEach(() => {
  accountLifecycle.invalidate();
  downloadText.mockReset();
  vi.restoreAllMocks();
});

describe('readWildcardFiles', () => {
  it('reads a text file as one wildcard named after it', async () => {
    await expect(read([file('colours.txt', 'red\n# a note\n\ngreen')], 'files')).resolves.toEqual([
      { name: 'colours', values: ['red', 'green'] },
    ]);
  });

  it('reads a nested yaml collection', async () => {
    await expect(read([file('all.yaml', 'animals:\n  dogs:\n    - corgi\n')], 'files')).resolves.toEqual([
      { name: 'animals/dogs', values: ['corgi'] },
    ]);
  });

  it('reads a json collection', async () => {
    await expect(read([file('all.json', '{"moods":["calm"]}')], 'files')).resolves.toEqual([
      { name: 'moods', values: ['calm'] },
    ]);
  });

  it('refuses a file that is not a supported format', async () => {
    await expect(read([file('README.md', '# Wildcards\n\nSome notes.')], 'files')).rejects.toBeInstanceOf(
      WildcardFileError
    );
  });

  it('refuses a collection too large to import instead of overflowing', async () => {
    const huge = JSON.stringify(Object.fromEntries(Array.from({ length: 25_000 }, (_, i) => [`w${i}`, ['v']])));

    await expect(read([file('huge.json', huge)], 'files')).rejects.toBeInstanceOf(WildcardFileError);
  });

  it('turns a folder`s nesting into `/` names', async () => {
    const picked = [inFolder('wildcards/animals/dogs.txt', 'corgi'), inFolder('wildcards/moods/dogs.txt', 'sleepy')];

    await expect(read(picked, 'folder')).resolves.toEqual([
      { name: 'animals/dogs', values: ['corgi'] },
      { name: 'moods/dogs', values: ['sleepy'] },
    ]);
  });
});

describe('isSupportedWildcardFile', () => {
  it('keeps the wildcard files and drops the rest', () => {
    const picked = [
      inFolder('wildcards/colours.txt', 'red'),
      inFolder('wildcards/all.yaml', 'a: [b]'),
      inFolder('wildcards/LICENSE', 'text'),
      inFolder('wildcards/README.md', '# notes'),
      inFolder('wildcards/preview.png', 'binary'),
      inFolder('wildcards/.DS_Store', 'binary'),
    ];

    expect(picked.filter(isSupportedWildcardFile).map((entry) => entry.name)).toEqual([
      'colours.txt',
      'all.yaml',
      'LICENSE',
    ]);
  });
});

it('does not download a wildcard export whose YAML serializer finishes after account rotation', async () => {
  let resolveStringify!: (value: string) => void;
  const serialized = new Promise<string>((resolve) => {
    resolveStringify = resolve;
  });
  const yaml = WILDCARD_COLLECTION_FORMATS.find((format) => format.id === 'yaml')!;
  vi.spyOn(yaml, 'stringify').mockReturnValue(serialized);
  const owner = accountLifecycle.activate('wildcard-export-a', ':user:wildcard-export-a');

  const exportPromise = downloadWildcards([{ name: 'colors', values: ['red'] }], 'yaml', owner);
  accountLifecycle.activate('wildcard-export-b', ':user:wildcard-export-b');
  resolveStringify('colors:\\n  - red\\n');

  await exportPromise.catch(() => undefined);

  expect(downloadText).not.toHaveBeenCalled();
});
