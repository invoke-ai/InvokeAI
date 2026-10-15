import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const axeScriptPath = require.resolve('axe-core/axe.min.js');

export const WCAG_RELEASE_TAGS = Object.freeze(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa']);

export const injectAxe = async (page) => {
  const isLoaded = await page.evaluate(() => typeof window.axe?.run === 'function');

  if (!isLoaded) {
    await page.addScriptTag({ path: axeScriptPath });
  }
};

export const runAxe = async (page, { exclude = [], include = [], runOnly = WCAG_RELEASE_TAGS, rules = {} } = {}) => {
  await injectAxe(page);

  return page.evaluate(
    ({ excludeSelectors, includeSelectors, ruleConfiguration, tags }) => {
      const context =
        includeSelectors.length > 0 || excludeSelectors.length > 0
          ? {
              exclude: excludeSelectors.map((selector) => [selector]),
              include: (includeSelectors.length > 0 ? includeSelectors : ['html']).map((selector) => [selector]),
            }
          : document;

      return window.axe.run(context, {
        resultTypes: ['violations'],
        rules: ruleConfiguration,
        runOnly: { type: 'tag', values: tags },
      });
    },
    {
      excludeSelectors: exclude,
      includeSelectors: include,
      ruleConfiguration: rules,
      tags: runOnly,
    }
  );
};

const formatNode = (node) => {
  const target = node.target.join(' ');
  const summary = node.failureSummary?.replaceAll(/\s+/g, ' ').trim();
  const html = node.html.replaceAll(/\s+/g, ' ').trim();

  return [`    ${target}`, `      ${html}`, summary ? `      ${summary}` : null].filter(Boolean).join('\n');
};

export const formatAxeViolations = (surface, violations) =>
  [
    `${surface} has ${violations.length} Axe violation${violations.length === 1 ? '' : 's'}:`,
    ...violations.flatMap((violation) => [
      `  [${violation.impact ?? 'unknown'}] ${violation.id}: ${violation.help} (${violation.helpUrl})`,
      ...violation.nodes.map(formatNode),
    ]),
  ].join('\n');

export const assertNoAxeViolations = async (page, surface, options) => {
  const results = await runAxe(page, options);

  if (results.violations.length > 0) {
    throw new Error(formatAxeViolations(surface, results.violations));
  }

  return results;
};
