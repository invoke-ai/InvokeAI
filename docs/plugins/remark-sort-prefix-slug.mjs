import path from 'node:path';

import { generateDocsId, getSortPrefix } from '../src/lib/sort-prefix.mjs';

/**
 * starlight-links-validator derives each page's URL from its file path and ignores the docs loader's
 * `generateId`, so it cannot find pages whose path carries a sorting prefix. It does honour a
 * frontmatter `slug`, so give those pages their real id as one while rendering.
 */
export function remarkSortPrefixSlug({ docsDir }) {
  return (_tree, file) => {
    const filePath = file.history[0];
    if (!filePath) return;
    const entry = path.relative(docsDir, filePath).split(path.sep).join('/');
    if (entry.startsWith('..') || !entry.split('/').some((segment) => getSortPrefix(segment) !== undefined)) return;

    file.data.astro ??= {};
    const frontmatter = (file.data.astro.frontmatter ??= {});
    if (typeof frontmatter.slug !== 'string') frontmatter.slug = generateDocsId({ entry, data: frontmatter });
  };
}
