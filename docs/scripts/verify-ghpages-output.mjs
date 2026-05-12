import { readFileSync } from 'node:fs';

const expectations = [
  {
    file: 'contributing/index.html',
    includes: ['href="/InvokeAI/contributing/new-contributor-guide/"'],
    excludes: ['href="/contributing/new-contributor-guide/"', 'newContributorChecklist.md'],
  },
  {
    file: 'contributing/contribution_guides/newContributorChecklist/index.html',
    includes: [
      'Redirecting to: /InvokeAI/contributing/new-contributor-guide',
      'content="0;url=/InvokeAI/contributing/new-contributor-guide"',
      'href="/InvokeAI/contributing/new-contributor-guide"',
    ],
    excludes: [
      'Redirecting to: /contributing/new-contributor-guide',
      'content="0;url=/contributing/new-contributor-guide"',
      'href="/contributing/new-contributor-guide"',
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
  throw new Error(`GitHub Pages output validation failed:\n- ${errors.join('\n- ')}`);
}

console.log('GitHub Pages output links look correct.');
