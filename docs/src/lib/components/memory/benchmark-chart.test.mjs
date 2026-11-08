import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { chartScale, formatMeasure, rowsInHtml, speedup } from './benchmark-chart.mjs';

describe('speedup', () => {
  it('matches the ratios measured for int8 Compute', () => {
    assert.equal(speedup(5.04, 2.93), 1.72);
    assert.equal(speedup(26.62, 20.71), 1.29);
    assert.equal(speedup(34.43, 9.04), 3.81);
  });

  it('rejects a time that is missing, zero or negative', () => {
    assert.throws(() => speedup(5, 0), /positive/);
    assert.throws(() => speedup(Number.NaN, 2), /positive/);
    assert.throws(() => speedup(-1, 2), /positive/);
  });
});

describe('rowsInHtml and chartScale', () => {
  const html =
    '<div class="bench-row" data-bench-row data-bench-label="FLUX.1 &quot;dev&quot;, 20 steps &amp; 1024²" ' +
    'data-bench-before="15.25" data-bench-after="7.85"><p>…</p></div>' +
    '<div data-bench-row data-bench-label="Krea-2" data-bench-before="21.64" data-bench-after="12.32"></div>';

  it('reads each row with its label unescaped', () => {
    assert.deepEqual(rowsInHtml(html), [
      { label: 'FLUX.1 "dev", 20 steps & 1024²', before: 15.25, after: 7.85 },
      { label: 'Krea-2', before: 21.64, after: 12.32 },
    ]);
  });

  it('scales to the slowest time of any row, before or after', () => {
    assert.equal(chartScale(rowsInHtml(html)), 21.64);
  });

  it('fails the build for a chart without rows or with a broken time', () => {
    assert.throws(() => rowsInHtml('<p>No rows</p>'), /at least one/);
    assert.throws(
      () => rowsInHtml('<div data-bench-row data-bench-label="x" data-bench-before="" data-bench-after="2"></div>'),
      /positive/
    );
  });
});

describe('formatMeasure', () => {
  it('keeps two decimals in the page language', () => {
    assert.equal(formatMeasure(1.7, 'en'), '1.70');
    assert.equal(formatMeasure(5.04, 'de'), '5,04');
  });
});
