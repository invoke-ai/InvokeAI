export function rehypePrefixBaseToRootLinks(options = {}) {
  const base = normalizeBase(options.base);

  return (tree) => {
    if (!base) {
      return;
    }

    walk(tree, (node) => {
      if (node.tagName !== 'a') {
        return;
      }

      const href = node.properties?.href;

      if (typeof href !== 'string') {
        return;
      }

      if (!href.startsWith('/') || href.startsWith('//') || href.startsWith(`${base}/`)) {
        return;
      }

      node.properties.href = `${base}${href}`;
    });
  };
}

function walk(node, visitor) {
  if (!node || typeof node !== 'object') {
    return;
  }

  if (node.type === 'element') {
    visitor(node);
  }

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
