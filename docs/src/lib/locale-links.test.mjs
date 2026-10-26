import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { localizeHtmlLinks } from './locale-links.mjs';

/** @param {string} href */
const anchor = (href) => `<a class="x" href="${href}">text</a>`;

describe('localizeHtmlLinks', () => {
  it('moves root-relative docs links into the locale, keeping fragments and other attributes', () => {
    const html = `<p>See ${anchor('/users-guide/gallery/introduction/#boards')} and <a href="/">home</a>.</p>`;
    assert.equal(
      localizeHtmlLinks(html, { base: '/', locale: 'de' }),
      `<p>See ${anchor('/de/users-guide/gallery/introduction/#boards')} and <a href="/de/">home</a>.</p>`
    );
  });

  it('keeps the base path in front of the locale', () => {
    assert.equal(
      localizeHtmlLinks(anchor('/InvokeAI/start-here/manual/'), { base: '/InvokeAI', locale: 'es' }),
      anchor('/InvokeAI/es/start-here/manual/')
    );
    assert.equal(localizeHtmlLinks(anchor('/InvokeAI'), { base: '/InvokeAI/', locale: 'es' }), anchor('/InvokeAI/es/'));
  });

  it('leaves root-only, localized, relative, external, protocol-relative and fragment links alone', () => {
    const hrefs = [
      '/download/',
      '/showcase/#submit',
      '/_astro/image.png',
      '/hi/configuration/docker/',
      'start-here/upgrading/',
      '../assets/a.png',
      'https://github.com/invoke-ai/InvokeAI',
      '//cdn.example.com/x.js',
      '#settings',
    ];
    const html = hrefs.map(anchor).join('');
    assert.equal(localizeHtmlLinks(html, { base: '/', locale: 'de' }), html);
  });

  it('leaves links outside the base path alone', () => {
    const html = anchor('/InvokeAIx/page/') + anchor('/other/page/');
    assert.equal(localizeHtmlLinks(html, { base: '/InvokeAI', locale: 'de' }), html);
  });

  it('reads quoted attribute values whole, whatever they contain', () => {
    // Expressive Code keeps `<` in `data-code` and escapes `"` only.
    const code = `<div data-code="<a href=&#x22;/x/&#x22;> <a href='/y/'>"><a title="a>b" href="/z/">z</a></div>`;
    assert.equal(
      localizeHtmlLinks(code, { base: '/', locale: 'de' }),
      `<div data-code="<a href=&#x22;/x/&#x22;> <a href='/y/'>"><a title="a>b" href="/de/z/">z</a></div>`
    );
    const lookalike = `<a title='see href="/x/"' href="/y/">y</a>`;
    assert.equal(
      localizeHtmlLinks(lookalike, { base: '/', locale: 'de' }),
      `<a title='see href="/x/"' href="/de/y/">y</a>`
    );
  });

  it('leaves script and style text alone', () => {
    const html = '<script>const a = `<a href="/x/">`;</script><style>a[href="/x/"]{}</style>';
    assert.equal(localizeHtmlLinks(html, { base: '/', locale: 'de' }), html);
  });

  it('only rewrites the href of anchors', () => {
    const html = '<img src="/users-guide/x.png"><a data-href="/users-guide/" href="/users-guide/">x</a>';
    assert.equal(
      localizeHtmlLinks(html, { base: '/', locale: 'de' }),
      '<img src="/users-guide/x.png"><a data-href="/users-guide/" href="/de/users-guide/">x</a>'
    );
  });
});
