import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { findSectionIcon } from './section-icon.mjs';

const sections = {
  'configuration/optimization': 'gauge',
  'guide/images/advanced': 'flask',
  'guide/models/local': 'cpu',
};
const topics = [
  { directory: 'configuration', icon: 'settings' },
  { directory: 'guide', icon: 'book' },
  { directory: 'guide/models', icon: 'box' },
];
const pages = { 'configuration/docker': 'container' };
const find = (path) => findSectionIcon(path, pages, sections, topics);

describe('findSectionIcon', () => {
  it('prefers a page\'s own icon', () => {
    assert.equal(find('/configuration/docker/'), 'container');
    assert.equal(find('/configuration/invokeai-yaml/'), 'settings');
  });

  it("uses the icon of the page's section, however deep the page is", () => {
    assert.equal(find('/configuration/optimization/low-vram-mode/'), 'gauge');
    assert.equal(find('guide/images/advanced/hidiffusion'), 'flask');
    // A section's index page is the section itself.
    assert.equal(find('/configuration/optimization/'), 'gauge');
  });

  it("falls back to the page's topic, the closest one first", () => {
    assert.equal(find('/configuration/external-database/'), 'settings');
    assert.equal(find('/guide/models/introduction/'), 'box');
    assert.equal(find('/guide/images/intro/'), 'book');
  });

  it('has no icon for pages outside every topic', () => {
    assert.equal(find('/download/'), undefined);
    assert.equal(find('/'), undefined);
  });
});
