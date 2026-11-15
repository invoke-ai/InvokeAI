/** Keep agent guidance and owner documents pointing at files and headings that exist. */

import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import test from 'node:test';

const packageRoot = resolve(import.meta.dirname, '..');

const withoutFences = (markdown) => markdown.replace(/^(```|~~~)[^\n]*\n[\s\S]*?^\1[^\n]*$/gm, '');

/** GitHub heading anchors: lowercase, punctuation dropped, spaces to hyphens, repeats suffixed. */
const headingAnchors = (markdown) => {
  const anchors = new Set();
  const seen = new Map();
  for (const [, heading] of withoutFences(markdown).matchAll(/^#{1,6}\s+(.+?)\s*#*\s*$/gm)) {
    const slug = heading
      .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/`/g, '')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s_-]/gu, '')
      .replace(/\s/g, '-');
    const count = seen.get(slug) ?? 0;
    seen.set(slug, count + 1);
    anchors.add(count === 0 ? slug : `${slug}-${count}`);
  }
  return anchors;
};

const relativeLinks = (markdown) =>
  [
    ...withoutFences(markdown)
      .replace(/`[^`\n]*`/g, '')
      .matchAll(/!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)/g),
  ]
    .map(([, target]) => target)
    .filter((target) => !/^[a-z][a-z\d+.-]*:/i.test(target));

const brokenLinks = (documentPath, markdown) =>
  relativeLinks(markdown).flatMap((link) => {
    const [path, anchor] = link.split('#', 2);
    const targetPath = path ? resolve(dirname(documentPath), decodeURIComponent(path)) : documentPath;
    if (!existsSync(targetPath)) {
      return [`${link}: no such file`];
    }
    if (anchor === undefined || !targetPath.endsWith('.md')) {
      return [];
    }
    const target = targetPath === documentPath ? markdown : readFileSync(targetPath, 'utf8');
    return headingAnchors(target).has(decodeURIComponent(anchor).toLowerCase()) ? [] : [`${link}: no such heading`];
  });

/** Package-level guidance plus the owner documents under src and performance. */
const guidanceDocuments = () => [
  ...readdirSync(packageRoot)
    .filter((name) => name.endsWith('.md'))
    .map((name) => join(packageRoot, name)),
  ...['src', 'performance'].flatMap((directory) =>
    readdirSync(join(packageRoot, directory), { recursive: true })
      .filter((name) => name.endsWith('.md'))
      .map((name) => join(packageRoot, directory, name))
  ),
];

test('derives GitHub anchors and ignores links inside code', () => {
  assert.deepEqual(
    [...headingAnchors('# Run `check:release` (CI)\n## Notes\n## Notes\n```md\n# Hidden\n```')],
    ['run-checkrelease-ci', 'notes', 'notes-1']
  );
  assert.deepEqual(relativeLinks('[a](A.md#x) ![b](img.png "t") [c](https://x.test) `[d](D.md)` [e](#local)'), [
    'A.md#x',
    'img.png',
    '#local',
  ]);
});

test('resolves every relative link and heading anchor in package guidance and owner documents', () => {
  const documents = guidanceDocuments();
  assert.ok(documents.some((path) => path.endsWith('VERIFICATION.md')));
  const broken = documents.flatMap((path) =>
    brokenLinks(path, readFileSync(path, 'utf8')).map((problem) => `${relative(packageRoot, path)}: ${problem}`)
  );
  assert.deepEqual(broken, [], 'Update the link to the moved file or renamed heading, or restore the target.');
});
