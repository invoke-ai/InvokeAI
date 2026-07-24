import { existsSync } from 'node:fs';
import { dirname, relative, resolve, sep } from 'node:path';

const relativeImportPattern = /(from\s+|import\s*)(['"])(\.\.?\/[^'"]+)\2/g;

export function remarkLocalizeContent(options = {}) {
  const locales = new Set(options.locales ?? []);

  return (tree, file) => {
    const context = getLocalizedFileContext(file?.path, locales);

    if (!context) {
      return;
    }

    walk(tree, (node) => {
      if (node.type === 'link' && typeof node.url === 'string') {
        node.url = localizeRootLink(node.url, context.locale);
      }

      if (node.type === 'image' && typeof node.url === 'string') {
        node.url = pointToSourceAsset(node.url, context);
      }

      if (node.type === 'mdxjsEsm' && typeof node.value === 'string') {
        node.value = node.value.replace(relativeImportPattern, (match, prefix, quote, specifier) => {
          const localizedSpecifier = pointToSourceAsset(specifier, context);
          return `${prefix}${quote}${localizedSpecifier}${quote}`;
        });
      }

      if (node.type === 'mdxJsxAttribute' && typeof node.value === 'string') {
        if (node.name === 'href' || node.name === 'link') {
          node.value = localizeRootLink(node.value, context.locale);
        } else if (node.name === 'src') {
          node.value = pointToSourceAsset(node.value, context);
        }
      }
    });
  };
}

function getLocalizedFileContext(filePath, locales) {
  if (typeof filePath !== 'string') {
    return undefined;
  }

  const normalizedPath = filePath.split(sep).join('/');
  const docsMarker = '/src/content/docs/';
  const markerIndex = normalizedPath.lastIndexOf(docsMarker);

  if (markerIndex === -1) {
    return undefined;
  }

  const docsRoot = normalizedPath.slice(0, markerIndex + docsMarker.length - 1);
  const pathWithinDocs = normalizedPath.slice(markerIndex + docsMarker.length);
  const [locale, ...sourceSegments] = pathWithinDocs.split('/');

  if (!locales.has(locale) || sourceSegments.length === 0) {
    return undefined;
  }

  return {
    locale,
    localizedDirectory: dirname(normalizedPath),
    sourceDirectory: dirname(resolve(docsRoot, ...sourceSegments)),
  };
}

function localizeRootLink(url, locale) {
  if (!url.startsWith('/') || url.startsWith('//')) {
    return url;
  }

  if (
    url === `/${locale}` ||
    url.startsWith(`/${locale}/`) ||
    url === '/download' ||
    url.startsWith('/download/')
  ) {
    return url;
  }

  return `/${locale}${url}`;
}

function pointToSourceAsset(url, context) {
  if (!url.startsWith('./') && !url.startsWith('../')) {
    return url;
  }

  const suffixIndex = url.search(/[?#]/);
  const pathPart = suffixIndex === -1 ? url : url.slice(0, suffixIndex);
  const suffix = suffixIndex === -1 ? '' : url.slice(suffixIndex);
  const sourceTarget = resolve(context.sourceDirectory, pathPart);

  if (!existsSync(sourceTarget)) {
    return url;
  }

  const localizedPath = relative(context.localizedDirectory, sourceTarget).split(sep).join('/');
  const normalizedPath = localizedPath.startsWith('.') ? localizedPath : `./${localizedPath}`;
  return `${normalizedPath}${suffix}`;
}

function walk(node, visitor) {
  if (!node || typeof node !== 'object') {
    return;
  }

  visitor(node);

  if (!Array.isArray(node.children)) {
    return;
  }

  for (const child of node.children) {
    walk(child, visitor);
  }
}

export const testing = {
  getLocalizedFileContext,
  localizeRootLink,
  pointToSourceAsset,
};
