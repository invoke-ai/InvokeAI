export function rehypePrefixBaseToRootLinks(options = {}) {
  const base = normalizeBase(options.base);

  return (tree) => {
    if (!base) {
      return;
    }

    walk(tree, (node) => {
      if (node.type === 'element') {
        if (node.tagName !== 'a') {
          return;
        }

        const href = node.properties?.href;

        if (typeof href === 'string') {
          node.properties.href = prefixBase(href, base);
        }

        return;
      }

      // MDX components such as Starlight's <LinkButton>/<LinkCard> stay JSX nodes all the way
      // through rehype rather than becoming `<a>` elements, and they pass `href` straight to the
      // DOM without applying the base. Without this branch, a root-relative href written in MDX
      // ships unprefixed and 404s on the GitHub Pages target.
      if (node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') {
        for (const attribute of node.attributes ?? []) {
          if (attribute?.type !== 'mdxJsxAttribute' || typeof attribute.value !== 'string') {
            continue;
          }

          if (attribute.name === 'href' || attribute.name === 'link') {
            attribute.value = prefixBase(attribute.value, base);
          }
        }
      }
    });
  };
}

function prefixBase(href, base) {
  if (!href.startsWith('/') || href.startsWith('//') || href === base || href.startsWith(`${base}/`)) {
    return href;
  }

  return `${base}${href}`;
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

function normalizeBase(base) {
  if (!base || base === '/') {
    return '';
  }

  return base.endsWith('/') ? base.slice(0, -1) : base;
}
