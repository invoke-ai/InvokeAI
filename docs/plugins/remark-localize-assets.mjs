import { statSync } from 'node:fs';
import path from 'node:path';

import { translatedLocales } from '../src/config/locales.mjs';

const relativePathRegEx = /^\.\.?\//;
const MODULE_SOURCE_TYPES = new Set(['ImportDeclaration', 'ExportNamedDeclaration', 'ExportAllDeclaration']);

/** @param {string} filePath */
const isFile = (filePath) => statSync(filePath, { throwIfNoEntry: false })?.isFile() ?? false;

/**
 * Crowdin adds translated pages to `src/content/docs/<locale>/` but not the images beside them, so a translated
 * page's relative image paths (inline and reference-style) and MDX import and export sources, copied from the English
 * page, would lead nowhere. Point each one at the English page's file, unless the translation ships its own. Paths
 * that are not files, such as relative page links, stay as they are. Frontmatter is out of reach: a translated page's
 * frontmatter cannot use relative image paths.
 * @param {{ docsDir: string }} options
 */
export function remarkLocalizeAssets({ docsDir }) {
  return (tree, file) => {
    const filePath = file.history[0];
    if (!filePath) return;
    const [locale, ...sourceSegments] = path.relative(docsDir, filePath).split(path.sep);
    if (!translatedLocales.includes(locale) || sourceSegments.length === 0) return;

    const localizedDirectory = path.dirname(filePath);
    const sourceDirectory = path.dirname(path.join(docsDir, ...sourceSegments));
    /** @param {string} url */
    const toSource = (url) => {
      if (!relativePathRegEx.test(url)) return url;
      const suffixIndex = url.search(/[?#]/);
      const filePart = suffixIndex === -1 ? url : url.slice(0, suffixIndex);
      const sourceTarget = path.resolve(sourceDirectory, filePart);
      if (isFile(path.resolve(localizedDirectory, filePart)) || !isFile(sourceTarget)) return url;
      const relative = path.relative(localizedDirectory, sourceTarget).split(path.sep).join('/');
      return `${relative}${suffixIndex === -1 ? '' : url.slice(suffixIndex)}`;
    };

    walk(tree, (node) => {
      if (node.type === 'image' || node.type === 'definition') {
        node.url = toSource(node.url);
      } else if (node.type === 'mdxjsEsm') {
        // MDX compiles imports and exports from the estree, not from the node's source text.
        for (const statement of node.data?.estree?.body ?? []) {
          if (!MODULE_SOURCE_TYPES.has(statement.type) || !statement.source) continue;
          const specifier = toSource(statement.source.value);
          if (specifier === statement.source.value) continue;
          statement.source.value = specifier;
          statement.source.raw = JSON.stringify(specifier);
        }
      }
    });
  };
}

/**
 * @param {{ type: string, children?: unknown[] } & Record<string, any>} node
 * @param {(node: any) => void} visitor
 */
function walk(node, visitor) {
  visitor(node);
  for (const child of node.children ?? []) walk(child, visitor);
}
