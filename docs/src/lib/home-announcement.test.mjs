import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { announcementHtml } from './home-announcement.mjs';

describe('announcementHtml', () => {
  it('links the text, escaped, so a translation cannot add markup', () => {
    const html = announcementHtml({ text: 'V7 <b>"jetzt"</b> & mehr', href: '/v7/de/start-here/upgrading/' });
    assert.match(html, /^<a class="home-announcement" href="\/v7\/de\/start-here\/upgrading\/"/);
    assert.ok(html.includes('aria-label="V7 &#60;b&#62;&#34;jetzt&#34;&#60;/b&#62; &#38; mehr"'));
    assert.ok(html.includes('</span>V7 &#60;b&#62;&#34;jetzt&#34;&#60;/b&#62; &#38; mehr</span>'));
    assert.ok(!html.includes('<b>'));
  });
});
