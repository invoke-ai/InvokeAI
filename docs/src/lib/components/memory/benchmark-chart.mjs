// @ts-check
// Numbers for the before/after timing chart (`BenchmarkChart.astro`, `BenchmarkRow.astro`): each row is one measured
// run, timed with a setting off (`before`) and on (`after`). Bars share one scale, the slowest time in the chart.

/**
 * How many times as fast the run is with the setting on, to two decimals: 5.04 s → 2.93 s is 1.72.
 * @param {number} before seconds with the setting off
 * @param {number} after seconds with the setting on
 */
export const speedup = (before, after) => {
  assertTime(before);
  assertTime(after);
  return Math.round((before / after) * 100) / 100;
};

/**
 * The rows of a rendered chart: each `BenchmarkRow` carries its label and times in `data-bench-label`,
 * `data-bench-before` and `data-bench-after`. A chart without rows, or with a broken time, is an authoring mistake and
 * fails the build.
 * @param {string} html the chart's rendered rows
 * @returns {{ label: string, before: number, after: number }[]}
 */
export const rowsInHtml = (html) => {
  const rows = [...html.matchAll(/<div\b[^>]*\bdata-bench-row\b[^>]*>/g)].map(([tag]) => {
    const attribute = (/** @type {string} */ name) =>
      unescapeHtml(tag.match(new RegExp(`\\b${name}="([^"]*)"`))?.[1] ?? '');
    const row = {
      label: attribute('data-bench-label'),
      before: Number(attribute('data-bench-before')),
      after: Number(attribute('data-bench-after')),
    };
    assertTime(row.before);
    assertTime(row.after);
    return row;
  });
  if (!rows.length) throw new Error('A BenchmarkChart needs at least one BenchmarkRow.');
  return rows;
};

/**
 * The chart's scale: the slowest time any row shows.
 * @param {{ before: number, after: number }[]} rows
 */
export const chartScale = (rows) => Math.max(...rows.flatMap((row) => [row.before, row.after]));

/** @param {string} text an HTML attribute value as rendered */
const unescapeHtml = (text) =>
  text
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'")
    .replaceAll('&#x27;', "'")
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&amp;', '&');

/**
 * A time or ratio in the page's language with exactly two decimals, as measured: `5.04`, `1.70`, or `5,04` in German.
 * @param {number} value
 * @param {string} lang a BCP 47 tag
 */
export const formatMeasure = (value, lang) =>
  new Intl.NumberFormat(lang, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);

/** @param {number} value */
function assertTime(value) {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`A benchmark time must be a positive number, not ${value}.`);
}
