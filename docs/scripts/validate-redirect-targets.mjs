import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const docsRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const contentRoot = join(docsRoot, 'src', 'content', 'docs');
const redirectsFile = join(docsRoot, 'src', 'config', 'redirects.ts');

const normalizeRoute = (route) => {
  const normalized = route
    .replace(/^\/+|\/+$/g, '')
    .split('/')
    .filter(Boolean)
    .map((segment) => segment.toLowerCase().replaceAll(' ', '-'))
    .join('/');

  return normalized ? `/${normalized}` : '/';
};

const collectDocsRoutes = (dir, routes = new Set()) => {
  for (const entry of readdirSync(dir)) {
    const entryPath = join(dir, entry);
    const stats = statSync(entryPath);

    if (stats.isDirectory()) {
      collectDocsRoutes(entryPath, routes);
      continue;
    }

    if (!entry.endsWith('.md') && !entry.endsWith('.mdx')) {
      continue;
    }

    const relativePath = relative(contentRoot, entryPath).replace(/\\/g, '/').replace(/\.mdx?$/, '');
    const route = relativePath.endsWith('/index') ? relativePath.slice(0, -'/index'.length) : relativePath;
    routes.add(normalizeRoute(route));

    const segments = route.split('/').filter(Boolean);
    for (let index = 1; index < segments.length; index++) {
      routes.add(normalizeRoute(segments.slice(0, index).join('/')));
    }
  }

  return routes;
};

if (!existsSync(contentRoot)) {
  throw new Error(`Docs content directory not found: ${contentRoot}`);
}

const redirectsSource = readFileSync(redirectsFile, 'utf8');
const redirectMatches = redirectsSource.matchAll(/^\s*['"]([^'"]+)['"]:\s*['"]([^'"]+)['"]/gm);
const redirectTargets = Array.from(redirectMatches, ([, from, to]) => ({ from, to }));
const docsRoutes = collectDocsRoutes(contentRoot);
const missingTargets = redirectTargets.filter(({ to }) => !docsRoutes.has(normalizeRoute(to)));

if (missingTargets.length > 0) {
  console.error('Redirect targets must resolve to generated docs routes:');
  for (const { from, to } of missingTargets) {
    console.error(`  ${from} -> ${to}`);
  }
  process.exit(1);
}

console.log(`Validated ${redirectTargets.length} redirect targets.`);
