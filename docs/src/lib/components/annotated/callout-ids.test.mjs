import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { calloutMarker, nextFigureIndex, numberCallouts } from './callout-ids.mjs';

const entry = (marker) => `<li id="${marker}"><a href="#${marker}"></a></li>`;

describe('numberCallouts', () => {
  it('numbers callouts in order, giving an entry and its pin the same id', () => {
    const html = entry(calloutMarker()) + entry(calloutMarker());
    assert.equal(
      numberCallouts(html, 2),
      '<li id="callout-2-1"><a href="#callout-2-1"></a></li><li id="callout-2-2"><a href="#callout-2-2"></a></li>'
    );
  });

  it('gives the same content the same ids on every render', () => {
    const render = () => numberCallouts(entry(calloutMarker()), 1);
    assert.equal(render(), render());
  });
});

describe('nextFigureIndex', () => {
  it('counts figures per page render', () => {
    const page = {};
    assert.equal(nextFigureIndex(page), 1);
    assert.equal(nextFigureIndex(page), 2);
    assert.equal(nextFigureIndex({}), 1);
  });
});
