import { existsSync, readdirSync, readFileSync } from 'node:fs';

import { translatedLocales } from '../src/config/locales.mjs';

const deployTarget = process.env.DEPLOY_TARGET ?? 'custom';
const base = deployTarget === 'ghpages' ? '/InvokeAI' : '';
const site = deployTarget === 'ghpages' ? 'https://invoke-ai.github.io' : 'https://v7.invoke.ai';
const withBase = (path) => `${base}${path}`;

/** @param {string} file a path inside dist */
const readDist = (file) => readFileSync(new URL(`../dist/${file}`, import.meta.url), 'utf8');

const expectations = [
  {
    file: 'index.html',
    includes: [
      `href="${withBase('/_astro/')}`,
      `src="${withBase('/_astro/')}`,
      `href="${withBase('/start-here/installation/')}`,
      // Header links: logo and version menu, main navigation, download.
      `href="${withBase('/')}"`,
      `href="${withBase('/showcase/')}"`,
      `href="${withBase('/download/')}`,
    ],
    excludes: deployTarget === 'custom' ? ['href="/InvokeAI/', 'src="/InvokeAI/'] : ['href="/_astro/', 'src="/_astro/'],
  },
  {
    file: 'contributing/index.html',
    includes: [
      `href="${withBase('/contributing/new-contributor-guide/')}`,
      // The header's Docs link is current on every docs page.
      `href="${withBase('/start-here/installation/')}" aria-current="true"`,
    ],
    excludes: [
      deployTarget === 'custom'
        ? 'href="/InvokeAI/contributing/new-contributor-guide/"'
        : 'href="/contributing/new-contributor-guide/"',
      'newContributorChecklist.md',
    ],
  },
  {
    // A locale's fallback page: content and header links stay in the locale, except to root-only releases. Search
    // engines are pointed at the English page it shows.
    file: 'de/contributing/index.html',
    includes: [
      `href="${withBase('/de/contributing/new-contributor-guide/')}`,
      `href="${withBase('/de/start-here/installation/')}" aria-current="true"`,
      `href="${withBase('/releases/')}"`,
      `<link rel="canonical" href="${site}${withBase('/contributing/')}"/>`,
    ],
    excludes: [`href="${withBase('/contributing/new-contributor-guide/')}`],
  },
  {
    file: 'contributing/contribution_guides/newContributorChecklist/index.html',
    includes: [
      `Redirecting to: ${withBase('/contributing/new-contributor-guide')}`,
      `content="0;url=${withBase('/contributing/new-contributor-guide')}`,
      `href="${withBase('/contributing/new-contributor-guide')}`,
    ],
    excludes: deployTarget === 'custom'
      ? [
          'Redirecting to: /InvokeAI/contributing/new-contributor-guide',
          'content="0;url=/InvokeAI/contributing/new-contributor-guide',
          'href="/InvokeAI/contributing/new-contributor-guide',
        ]
      : [
          'Redirecting to: /contributing/new-contributor-guide',
          'content="0;url=/contributing/new-contributor-guide',
          'href="/contributing/new-contributor-guide',
        ],
  },
];

const errors = [];

for (const { file, includes = [], excludes = [] } of expectations) {
  const html = readDist(file);

  for (const expected of includes) {
    if (!html.includes(expected)) {
      errors.push(`${file} is missing ${expected}`);
    }
  }

  for (const unexpected of excludes) {
    if (html.includes(unexpected)) {
      errors.push(`${file} still contains ${unexpected}`);
    }
  }
}

// starlight-markdown serves each docs page's Markdown source, which the page's "Copy page" and "View as Markdown"
// actions point at.
const markdownPage = '/start-here/installation/';
const markdown = readDist(`${markdownPage.slice(1)}index.md`);
if (!markdown.startsWith('# ') || markdown.includes('<html')) {
  errors.push(`dist${markdownPage}index.md is not the page's Markdown source`);
}
// A locale's fallback page shows the English page, so its actions use the English source.
for (const page of [markdownPage, `/de${markdownPage}`]) {
  if (!readDist(`${page.slice(1)}index.html`).includes(`data-markdown-href="${withBase(`${markdownPage}index.md`)}"`)) {
    errors.push(`${page.slice(1)}index.html does not link its page actions to ${withBase(`${markdownPage}index.md`)}`);
  }
}

// starlight-changelogs builds release pages at the root only (`astro.config.mjs`).
if (existsSync(new URL('../dist/de/releases', import.meta.url))) {
  errors.push('dist/de/releases exists, but release pages are root-only');
}

// Language alternates list translations only, so pages that are never translated (the contributor docs stay English)
// and pages without locale routes list none.
const untranslatedFiles = [
  'contributing/index.html',
  'de/contributing/index.html',
  '404.html',
  'releases/index.html',
  'download/index.html',
];
for (const file of untranslatedFiles) {
  if (readDist(file).includes('<link rel="alternate" hreflang=')) {
    errors.push(`${file} lists language alternates, but has no translation`);
  }
}

/** @param {string} directory a directory inside dist */
const htmlFilesIn = (directory) =>
  readdirSync(new URL(`../dist/${directory}/`, import.meta.url), { recursive: true })
    .filter((file) => file.endsWith('.html'))
    .map((file) => `${directory}/${file}`);

// Every internal link on the hand-written pages, and on every page of a locale, leads to a built file. The links
// validator only checks Markdown links, and a locale's pages move links into the locale after rendering, where links
// to root-only pages must stay at the root.
/** @type {Map<string, boolean>} */
const builtFiles = new Map();
const isBuilt = (target) => {
  if (!builtFiles.has(target)) builtFiles.set(target, existsSync(new URL(`../dist${target}`, import.meta.url)));
  return builtFiles.get(target);
};
const linkCheckedFiles = [
  'index.html',
  'download/index.html',
  'showcase/index.html',
  ...translatedLocales.flatMap(htmlFilesIn),
];
for (const file of linkCheckedFiles) {
  const pageUrl = new URL(`${base}/${file.replace(/index\.html$/, '')}`, 'https://site.invalid');
  for (const [, href] of readDist(file).matchAll(/\shref="([^"]*)"/g)) {
    const url = new URL(href.replaceAll('&amp;', '&'), pageUrl);
    if (url.origin !== pageUrl.origin || (url.pathname === pageUrl.pathname && url.hash)) continue;
    if (!url.pathname.startsWith(`${base}/`)) {
      errors.push(`${file} links to ${href}, outside the ${base || '/'} base path`);
      continue;
    }
    const path = decodeURIComponent(url.pathname.slice(base.length));
    const target = path.endsWith('/') ? `${path}index.html` : /\.[^/]+$/.test(path) ? path : `${path}/index.html`;
    if (!isBuilt(target)) {
      errors.push(`${file} links to ${href}, but dist has no ${target.slice(1)}`);
    }
  }
}

if (errors.length > 0) {
  throw new Error(`${deployTarget} output validation failed:\n- ${errors.join('\n- ')}`);
}

console.log(`${deployTarget} output links and assets look correct.`);
