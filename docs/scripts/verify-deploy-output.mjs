import { readFileSync } from 'node:fs';

const deployTarget = process.env.DEPLOY_TARGET ?? 'custom';
const base = deployTarget === 'ghpages' ? '/InvokeAI' : '';
const withBase = (path) => `${base}${path}`;
const siteUrl = (path) =>
  deployTarget === 'ghpages' ? `https://invoke-ai.github.io${base}${path}` : `https://invoke.ai${path}`;

const expectations = [
  {
    file: 'index.html',
    includes: [
      `href="${withBase('/_astro/')}`,
      `src="${withBase('/_astro/')}`,
      `href="${withBase('/start-here/installation/')}`,
    ],
    excludes: deployTarget === 'custom' ? ['href="/InvokeAI/', 'src="/InvokeAI/'] : ['href="/_astro/', 'src="/_astro/'],
  },
  {
    file: 'contributing/index.html',
    includes: [`href="${withBase('/contributing/new-contributor-guide/')}`],
    excludes: [
      deployTarget === 'custom'
        ? 'href="/InvokeAI/contributing/new-contributor-guide/"'
        : 'href="/contributing/new-contributor-guide/"',
      'newContributorChecklist.md',
    ],
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

for (const locale of ['de', 'es', 'hi']) {
  expectations.push({
    file: `${locale}/start-here/installation/index.html`,
    includes: [
      `<html lang="${locale}"`,
      'lang="en" dir="ltr"',
      `hreflang="${locale}" href="${siteUrl(`/${locale}/start-here/installation/`)}`,
      `href="${withBase(`/${locale}/start-here/system-requirements/`)}`,
      'href="https://crowdin.com/project/invokeai-docs"',
      'data-pagefind-body',
    ],
    excludes: [
      `href="${withBase(`/${locale}/${locale}/`)}`,
      `href="${withBase('/start-here/system-requirements/')}"`,
    ],
  });
}

const errors = [];

for (const { file, includes = [], excludes = [] } of expectations) {
  const html = readFileSync(new URL(`../dist/${file}`, import.meta.url), 'utf8');

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

if (errors.length > 0) {
  throw new Error(`${deployTarget} output validation failed:\n- ${errors.join('\n- ')}`);
}

console.log(`${deployTarget} output links and assets look correct.`);
