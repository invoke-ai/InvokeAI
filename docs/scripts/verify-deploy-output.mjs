import { readFileSync } from 'node:fs';

const deployTarget = process.env.DEPLOY_TARGET ?? 'custom';
const base = deployTarget === 'ghpages' ? '/InvokeAI' : '';
const withBase = (path) => `${base}${path}`;

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
