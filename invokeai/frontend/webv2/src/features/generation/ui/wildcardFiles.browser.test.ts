import { isSupportedWildcardFile, readWildcardFiles, WildcardFileError } from '@features/generation/ui/wildcardFiles';
import { describe, expect, it } from 'vitest';

const file = (name: string, contents: string) => new File([contents], name, { type: 'text/plain' });

/** What a directory pick hands over: a name plus the path relative to the root. */
const inFolder = (path: string, contents: string): File => {
  const picked = file(path.slice(path.lastIndexOf('/') + 1), contents);

  Object.defineProperty(picked, 'webkitRelativePath', { value: path });

  return picked;
};

describe('readWildcardFiles', () => {
  it('reads a text file as one wildcard named after it', async () => {
    await expect(readWildcardFiles([file('colours.txt', 'red\n# a note\n\ngreen')], 'files')).resolves.toEqual([
      { name: 'colours', values: ['red', 'green'] },
    ]);
  });

  it('reads a nested yaml collection', async () => {
    await expect(readWildcardFiles([file('all.yaml', 'animals:\n  dogs:\n    - corgi\n')], 'files')).resolves.toEqual([
      { name: 'animals/dogs', values: ['corgi'] },
    ]);
  });

  it('reads a json collection', async () => {
    await expect(readWildcardFiles([file('all.json', '{"moods":["calm"]}')], 'files')).resolves.toEqual([
      { name: 'moods', values: ['calm'] },
    ]);
  });

  it('takes an extensionless file as text', async () => {
    await expect(readWildcardFiles([file('colours', 'red')], 'files')).resolves.toEqual([
      { name: 'colours', values: ['red'] },
    ]);
  });

  // `accept` on a file input is a hint, not a rule, and a wildcard folder
  // routinely has a readme sitting in it. Read as text, that imported as a
  // wildcard named `README` whose values were the markdown.
  it('refuses a file that is not a supported format', async () => {
    await expect(readWildcardFiles([file('README.md', '# Wildcards\n\nSome notes.')], 'files')).rejects.toBeInstanceOf(
      WildcardFileError
    );
  });

  it('names the file it could not read', async () => {
    await expect(readWildcardFiles([file('broken.json', '{ nope')], 'files')).rejects.toMatchObject({
      fileName: 'broken.json',
    });
  });

  // Regression: `parsed.push(...entries)` is a call with one argument per entry,
  // which overflows the argument stack — at a threshold that differs by engine,
  // and reported as though the file had failed to parse.
  it('refuses a collection too large to import instead of overflowing', async () => {
    const huge = JSON.stringify(Object.fromEntries(Array.from({ length: 25_000 }, (_, i) => [`w${i}`, ['v']])));

    await expect(readWildcardFiles([file('huge.json', huge)], 'files')).rejects.toBeInstanceOf(WildcardFileError);
  });

  // The whole point of picking a folder: a1111 nests wildcards in directories,
  // and only the relative path says that `dogs.txt` is `animals/dogs`. Picking
  // the two files by hand gives two wildcards both called `dogs`.
  it('turns a folder`s nesting into `/` names', async () => {
    const picked = [inFolder('wildcards/animals/dogs.txt', 'corgi'), inFolder('wildcards/moods/dogs.txt', 'sleepy')];

    await expect(readWildcardFiles(picked, 'folder')).resolves.toEqual([
      { name: 'animals/dogs', values: ['corgi'] },
      { name: 'moods/dogs', values: ['sleepy'] },
    ]);
  });

  it('retains direct file names even when a relative path is present', async () => {
    const picked = inFolder('ignored/animals/dogs.txt', 'corgi');

    await expect(readWildcardFiles([picked], 'files')).resolves.toEqual([{ name: 'dogs', values: ['corgi'] }]);
  });
});

describe('isSupportedWildcardFile', () => {
  // A folder pick hands over everything in it, so the ones that are not
  // wildcards get dropped rather than failing the whole import.
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
