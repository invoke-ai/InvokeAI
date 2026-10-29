import assert from 'node:assert/strict';
import { cpSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { checkTranslations } from './translation-checks.mjs';

const docsRoot = fileURLToPath(new URL('../../', import.meta.url));

const frontmatter = (fields) => `---\n${fields}\n---\n`;

const englishFrontmatter = `title: Organizing your media
description: Use the Gallery to organize images.
template: splash
hero:
  title: Your gallery
  tagline: Boards for everything.
  actions:
    - text: Install
      link: /start-here/installation/
      icon: right-arrow
sidebar:
  order: 1
lastUpdated: 2026-10-19`;

const germanFrontmatter = `title: Medien organisieren
description: Mit der Galerie Bilder organisieren.
template: splash
hero:
  title: Deine Galerie
  tagline: Boards für alles.
  actions:
    - text: Installieren
      link: /start-here/installation/
      icon: right-arrow
sidebar:
  order: 1
lastUpdated: 2026-10-19`;

const imports = `import { Image } from 'astro:assets';
import { LinkCard, Steps } from '@astrojs/starlight/components';
import overview from '../assets/overview.webp';
`;

const englishBody = `
${imports}
The **Gallery** holds every image. Open the [Gallery settings](/users-guide/gallery/#gallery-settings) to change it.

{/* Keep this paragraph short. */}

<Image src={overview} alt="The Gallery" width={420} />

<LinkCard title="Install" href="/start-here/installation/" description="Set up Invoke" />

See <a href="/start-here/">Start here</a>.

:::note
Boards are shared between projects.
:::

:::tip[Good to know]{icon="star"}
Starred items stay at the top.
:::

<Steps>
  1. Open the board menu.

      \`\`\`sh
      invokeai-web --root ~/invokeai
      \`\`\`
</Steps>

Run \`invokeai-web\` to start.
`;

// Reordered sentences, translated link text, comment and aside label, German quotes, a translated heading fragment,
// colons that parse as text directives, and code samples whose braces, tags and backticks differ from English.
const germanBody = `
${imports}
Die „Galerie“ enthält jedes Bild. Ändern lässt sie sich in den
[Galerie-Einstellungen](/users-guide/gallery/#galerie-einstellungen).

{/* Diesen Absatz kurz halten. */}

<Image src={overview} alt="Die Galerie" width={420} />

<LinkCard title="Installieren" href="/start-here/installation/" description="Invoke einrichten" />

Siehe <a href="/start-here/">Erste Schritte</a>.

:::note
Boards gehören allen Projekten, siehe:Galerie (Stand 10:30 Uhr).
:::

:::tip[Gut zu wissen]{icon="star"}
Markierte Elemente bleiben oben.
:::

<Steps>
  1. Öffne das Board-Menü.

      \`\`\`\`sh
      # Beispiel: {root}, <Pfad> und \`\`\` sind hier nur Text
      invokeai-web --root ~/invokeai
      \`\`\`\`
</Steps>

Starte \`invokeai-web {--root}\` mit \`\`<Pfad> \` \`\`.
`;

const englishPage = frontmatter(englishFrontmatter) + englishBody;
const germanPage = frontmatter(germanFrontmatter) + germanBody;

const englishMarkdown = `---
title: Prompt tools
---

Press <kbd>Ctrl</kbd> to open the [tools](/users-guide/prompt-tools/). Braces like {this} are text.
`;

const germanMarkdown = `---
title: Prompt-Werkzeuge
---

Mit <kbd>Strg</kbd> öffnest du die [Werkzeuge](/users-guide/prompt-tools/). Klammern wie {diese} sind Text.
`;

const englishMessages = {
  'page.updated': 'Updated <time>{{date}}</time>',
  'search.resultCount_one': '{{count}} result',
  'search.resultCount_other': '{{count}} results',
  'nav.siteNavLabel': 'Main',
};

const germanMessages = {
  'page.updated': 'Aktualisiert am <time>{{date}}</time>',
  'search.resultCount_one': 'ein Ergebnis',
  'search.resultCount_other': '{{count}} Ergebnisse',
  'nav.siteNavLabel': 'Haupt',
};

const pageFile = 'src/content/docs/de/Users Guide/06.Gallery/introduction.mdx';
const markdownFile = 'src/content/docs/de/Users Guide/04.Image Generation/prompt-tools.md';

describe('checkTranslations', () => {
  let root = '';

  /** @param {string} file a path from the docs package directory */
  const write = (file, content) => {
    mkdirSync(path.dirname(path.join(root, file)), { recursive: true });
    writeFileSync(path.join(root, file), typeof content === 'string' ? content : JSON.stringify(content));
  };
  const writeGermanMessages = (messages) => write('src/content/i18n/de.json', { ...germanMessages, ...messages });

  beforeEach(() => {
    root = mkdtempSync(path.join(tmpdir(), 'translation-checks-'));
    write('src/content/docs/index.mdx', '---\ntitle: Home\n---\n');
    write('src/content/docs/assets/logo.webp', '');
    write('src/content/docs/Users Guide/06.Gallery/introduction.mdx', englishPage);
    write('src/content/docs/Users Guide/04.Image Generation/prompt-tools.md', englishMarkdown);
    write('src/content/i18n/en.json', englishMessages);
    write(pageFile, germanPage);
    write(markdownFile, germanMarkdown);
    writeGermanMessages({});
    write('src/content/i18n/es.json', {
      'search.resultCount_one': '{{count}} resultado',
      'search.resultCount_many': '{{count}} de resultados',
      'search.resultCount_other': '{{count}} resultados',
    });
  });

  afterEach(() => rmSync(root, { recursive: true, force: true }));

  it('passes the docs as they are', () => {
    assert.deepEqual(checkTranslations(docsRoot), []);
  });

  it('passes faithful translations, whatever their wording, comments, code samples and plural forms', () => {
    assert.deepEqual(checkTranslations(root), []);
  });

  it('passes a translation with a byte order mark and CRLF line endings', () => {
    write(pageFile, `﻿${germanPage.replace(/\n/g, '\r\n')}`);
    assert.deepEqual(checkTranslations(root), []);
  });

  it('passes an unchanged copy of every English page', () => {
    const pages = readdirSync(path.join(docsRoot, 'src/content/docs'), { recursive: true, withFileTypes: true })
      .filter((entry) => entry.isFile() && /\.mdx?$/.test(entry.name))
      .map((entry) => path.relative(path.join(docsRoot, 'src/content/docs'), path.join(entry.parentPath, entry.name)));
    assert.ok(pages.length > 50);
    for (const page of pages) {
      const source = path.join(docsRoot, 'src/content/docs', page);
      cpSync(source, path.join(root, 'src/content/docs', page));
      cpSync(source, path.join(root, 'src/content/docs/de', page));
    }
    assert.deepEqual(checkTranslations(root), []);
  });

  it('flags locales that locales.mjs does not configure', () => {
    write('src/content/docs/fr/start-here/manual.mdx', '---\ntitle: Manuel\n---\n');
    write('src/content/i18n/fr.json', {});
    write('src/content/i18n/notes.txt', '');
    const problems = checkTranslations(root);
    assert.equal(problems.length, 3);
    assert.match(problems[0], /^src\/content\/docs\/fr\/: neither a configured locale/);
    assert.match(problems[1], /^src\/content\/i18n\/(fr\.json|notes\.txt): not en\.json/);
    assert.match(problems[2], /^src\/content\/i18n\/(fr\.json|notes\.txt): not en\.json/);
  });

  it('flags translated pages without an English page, and files that are not pages', () => {
    write('src/content/docs/de/start-here/removed.mdx', '---\ntitle: Entfernt\n---\n');
    write('src/content/docs/de/Users Guide/06.Gallery/introduction.markdown', '# Galerie\n');
    write('src/content/docs/de/Users Guide/assets/overview.svg', '<svg onload="alert(1)" />');
    assert.deepEqual(checkTranslations(root).sort(), [
      'src/content/docs/de/Users Guide/06.Gallery/introduction.markdown: no English page at ' +
        'src/content/docs/Users Guide/06.Gallery/introduction.markdown.',
      'src/content/docs/de/Users Guide/assets/overview.svg: not a page. Crowdin writes only translated pages here.',
      'src/content/docs/de/start-here/removed.mdx: no English page at src/content/docs/start-here/removed.mdx.',
    ]);
  });

  /** Each edit of the German page, and the problems it must raise, in order. */
  const pageViolations = [
    // Code and markup the English page lacks.
    {
      name: 'a changed import',
      edit: (page) => page.replace("'../assets/overview.webp'", "'https://example.com/x.js'"),
      problems: [
        /adds the import or export code `import overview from 'https:\/\/example\.com\/x\.js';`/,
        /lacks the English page's `import overview from '\.\.\/assets\/overview\.webp';`/,
      ],
    },
    {
      name: 'code glued to an import',
      edit: (page) => page.replace("overview.webp';\n", "overview.webp';\nalert(document.cookie);\n"),
      problems: [/cannot be parsed: Unexpected `ExpressionStatement` in code: only import\/exports are supported/],
    },
    {
      name: 'a new export',
      edit: (page) => `${page}\nexport const leak = globalThis.process;\n`,
      problems: [/adds the import or export code `export const leak = globalThis\.process;`/],
    },
    {
      name: 'a new component',
      edit: (page) => page.replace('jedes Bild.', 'jedes Bild. <Aside>Hinweis</Aside>'),
      problems: [/adds `<Aside>`, which the English page does not use/],
    },
    {
      name: 'a new expression',
      edit: (page) => page.replace('jedes Bild.', 'jedes Bild {globalThis.secret}.'),
      problems: [/adds the expression `\{globalThis\.secret\}`/],
    },
    {
      name: 'a changed attribute expression',
      edit: (page) => page.replace('width={420}', 'width={fetch("https://example.com")}'),
      problems: [/adds the expression `<Image width=\{fetch\("https:\/\/example\.com"\)\}>`/],
    },
    {
      name: 'a spread attribute',
      edit: (page) => page.replace('<Image src={overview}', '<Image {...overview}'),
      problems: [/adds the expression `<Image \{\.\.\.overview\}>`/],
    },
    {
      name: 'a changed attribute value',
      edit: (page) => page.replace('<a href="/start-here/">', '<a href="/start-here/" style="position:fixed">'),
      problems: [/adds `<a style="position:fixed">`/],
    },
    // Code hidden from a text scan by backticks and fences that are not code to the MDX parser.
    {
      name: 'an expression between backticks in separate paragraphs',
      edit: (page) =>
        page.replace('jedes Bild.', 'jedes ` Bild.\n\n{globalThis.process.env.GITHUB_TOKEN}\n\nNoch ein ` hier.'),
      problems: [/adds the expression `\{globalThis\.process\.env\.GITHUB_TOKEN\}`/],
    },
    {
      name: 'an expression after an escaped backtick',
      edit: (page) => page.replace('jedes Bild.', 'jedes \\` {alert(1)} ` Bild.'),
      problems: [/adds the expression `\{alert\(1\)\}`/],
    },
    {
      name: 'an expression between backticks in attribute strings',
      edit: (page) =>
        page.replace(
          '<LinkCard title="Installieren" href="/start-here/installation/" description="Invoke einrichten" />',
          '<LinkCard title="`" href={globalThis.x} description="`" />'
        ),
      problems: [/adds the expression `<LinkCard href=\{globalThis\.x\}>`/],
    },
    {
      name: 'an expression after a fence whose info string has a backtick',
      edit: (page) => `${page}\n\`\`\` a\`b\n{alert(1)}\n\`\`\`\n`,
      problems: [/adds the expression `\{alert\(1\)\}`/],
    },
    {
      name: 'an export after a fence that a list item closes',
      edit: (page) => `${page}\n- Punkt\n  \`\`\`\nexport const leak = globalThis.process.env;\n\n`,
      problems: [/adds the import or export code `export const leak = globalThis\.process\.env;`/],
    },
    {
      name: 'an expression after a fence that an aside closes',
      edit: (page) => `${page}\n:::note\n\`\`\`\n:::\n{globalThis.process.env.GITHUB_TOKEN}\n\n\`\`\`\n`,
      problems: [/adds the expression `\{globalThis\.process\.env\.GITHUB_TOKEN\}`/],
    },
    {
      name: 'an export after a fence that an aside closes',
      edit: (page) => `${page}\n:::note\n\`\`\`\n:::\nexport const leak = globalThis.process.env;\n\n\`\`\`\n`,
      problems: [/adds the import or export code `export const leak = globalThis\.process\.env;`/],
    },
    {
      name: 'a new aside type',
      edit: (page) => page.replace(':::note\nBoards', ':::danger\nBoards'),
      problems: [/adds `:::danger`, which the English page does not use/],
    },
    {
      name: 'a changed directive attribute',
      edit: (page) => page.replace('{icon="star"}', '{icon="rocket"}'),
      problems: [/adds `:::tip\{icon="rocket"\}`/],
    },
    // Destinations and markup written as character references.
    {
      name: 'a javascript: link written with &colon;',
      edit: (page) => page.replace('jedes Bild.', 'jedes [Bild](javascript&colon;alert(1)).'),
      problems: [/links to `javascript:alert\(1\)`/],
    },
    {
      name: 'a javascript: href written with &#58;',
      edit: (page) => page.replace('<a href="/start-here/">', '<a href="javascript&#58;alert(1)">'),
      problems: [/adds `<a href="javascript:alert\(1\)">`/],
    },
    {
      name: 'markup in a card title written with entities',
      edit: (page) => page.replace('title="Installieren"', 'title="&lt;img src=x onerror&#61;alert(1)&gt;"'),
      problems: [/`<LinkCard title>` contains "<"/],
    },
    {
      name: 'a new link destination',
      edit: (page) => page.replace('jedes Bild.', 'jedes [Bild](https://tracker.example/).'),
      problems: [/links to `https:\/\/tracker\.example\/`/],
    },
    {
      name: 'a new autolinked URL',
      edit: (page) => page.replace('jedes Bild.', 'jedes Bild, siehe https://tracker.example/x.'),
      problems: [/links to `https:\/\/tracker\.example\/x`/],
    },
    {
      name: 'HTML syntax MDX cannot compile',
      edit: (page) => page.replace('jedes Bild.', 'jedes Bild. <svg onload=alert(1)>'),
      problems: [/cannot be parsed: Unexpected character/],
    },
    // Frontmatter beyond prose.
    {
      name: 'a head entry',
      edit: (page) =>
        page.replace('order: 1', 'order: 1\nhead:\n  - tag: script\n    content: alert(1)'),
      problems: [/frontmatter `head` is not in the English page/],
    },
    {
      name: 'a changed hero link written with a YAML escape',
      edit: (page) => page.replace('link: /start-here/installation/', 'link: "javascript\\x3aalert(1)"'),
      problems: [/frontmatter `hero\.actions\.0\.link` differs from the English page/],
    },
    {
      name: 'a changed non-prose value',
      edit: (page) => page.replace('  order: 1', '  order: 2'),
      problems: [/frontmatter `sidebar\.order` differs from the English page/],
    },
    {
      name: 'markup in the hero title written with a YAML escape',
      edit: (page) => page.replace('  title: Deine Galerie', '  title: "\\x3cimg src=x onerror\\x3dalert(1)>"'),
      problems: [/frontmatter `hero\.title` contains "<"/],
    },
    {
      name: 'markup in frontmatter after a leading blank line',
      edit: (page) => `\n${page.replace('tagline: Boards für alles.', 'tagline: Boards <b>für</b> alles.')}`,
      problems: [/frontmatter `hero\.tagline` contains "<"/],
    },
    {
      name: 'an expression after a closing delimiter with a trailing space',
      edit: (page) => page.replace('2026-10-19\n---\n', '2026-10-19\n--- \n\n{globalThis.secret}\n'),
      problems: [/adds the expression `\{globalThis\.secret\}`/],
    },
    {
      name: 'a head entry in TOML frontmatter',
      edit: (page) =>
        page.replace(
          frontmatter(germanFrontmatter),
          `+++
title = "Medien organisieren"
description = "Mit der Galerie Bilder organisieren."
template = "splash"
lastUpdated = 2026-10-19T00:00:00Z

[hero]
title = "Deine Galerie"
tagline = "Boards für alles."

[[hero.actions]]
text = "Installieren"
link = "/start-here/installation/"
icon = "right-arrow"

[sidebar]
order = 1

[[head]]
tag = "script"
content = "alert(1)"
+++
`
        ),
      problems: [/frontmatter `head` is not in the English page/],
    },
    {
      name: 'frontmatter that is not valid YAML',
      edit: (page) => page.replace('title: Medien organisieren', 'title: [Medien'),
      problems: [/cannot be parsed: /],
    },
  ];

  for (const { name, edit, problems: expected } of pageViolations) {
    it(`flags ${name} in a translated page`, () => {
      write(pageFile, edit(germanPage));
      const problems = checkTranslations(root);
      assert.equal(problems.length, expected.length, problems.join('\n'));
      expected.forEach((pattern, index) => {
        assert.ok(problems[index].startsWith(`${pageFile}: `), problems[index]);
        assert.match(problems[index], pattern);
      });
    });
  }

  const markdownViolations = [
    {
      name: 'raw HTML',
      edit: (page) => page.replace('{diese}', '<iframe src="https://example.com">'),
      problem: /adds the HTML `<iframe src="https:\/\/example\.com">`/,
    },
    {
      name: 'a javascript: autolink',
      edit: (page) => page.replace('{diese}', '<javascript:alert(1)>'),
      problem: /links to `javascript:alert\(1\)`/,
    },
    {
      name: 'a data: link',
      edit: (page) => page.replace('{diese}', '[x](data:text/html;base64,PHNjcmlwdD4=)'),
      problem: /links to `data:text\/html;base64,PHNjcmlwdD4=`/,
    },
  ];

  for (const { name, edit, problem } of markdownViolations) {
    it(`flags ${name} in a translated Markdown page`, () => {
      write(markdownFile, edit(germanMarkdown));
      const problems = checkTranslations(root);
      assert.equal(problems.length, 1, problems.join('\n'));
      assert.ok(problems[0].startsWith(`${markdownFile}: `), problems[0]);
      assert.match(problems[0], problem);
    });
  }

  const messageViolations = [
    { name: 'a key en.json lacks', messages: { 'search.title': 'Suche' }, problem: '"search.title" is not in en.json' },
    {
      name: 'a plural form of a message without plurals',
      messages: { 'nav.siteNavLabel_one': 'Haupt' },
      problem: '"nav.siteNavLabel_one" is not in en.json',
    },
    {
      name: 'a renamed placeholder',
      messages: { 'page.updated': 'Aktualisiert <time>{{datum}}</time>' },
      problem: '"page.updated" has placeholders English does not: {{datum}}.',
      alsoReports: '"page.updated" is missing placeholders: {{date}}.',
    },
    {
      name: 'a dropped placeholder',
      messages: { 'page.updated': 'Aktualisiert <time>heute</time>' },
      problem: '"page.updated" is missing placeholders: {{date}}.',
    },
    {
      name: 'a renamed tag',
      messages: { 'page.updated': 'Aktualisiert <zeit>{{date}}</zeit>' },
      problem: '"page.updated" has tags English does not: <zeit>.',
      alsoReports: '"page.updated" is missing tags: <time>.',
    },
    {
      name: 'a placeholder no English plural form has',
      messages: { 'search.resultCount_few': '{{count}} von {{total}} Ergebnissen' },
      problem: '"search.resultCount_few" has placeholders English does not: {{total}}.',
    },
    {
      name: 'a script',
      messages: { 'nav.siteNavLabel': 'Haupt<script src=//example.com/x.js>' },
      problem: '"nav.siteNavLabel" contains <script>.',
    },
    {
      name: 'an event handler',
      messages: { 'nav.siteNavLabel': '<img src=x onerror=alert(1)>' },
      problem: '"nav.siteNavLabel" contains an on… event handler attribute.',
    },
    {
      name: 'a javascript: URL',
      messages: { 'nav.siteNavLabel': 'javascript:alert(1)' },
      problem: '"nav.siteNavLabel" contains a javascript: URL.',
    },
    {
      name: 'a value that is not a string',
      messages: { 'nav.siteNavLabel': 1 },
      problem: '"nav.siteNavLabel" is not a string.',
    },
  ];

  for (const { name, messages, problem, alsoReports } of messageViolations) {
    it(`flags ${name} in a locale's UI strings`, () => {
      writeGermanMessages(messages);
      const expected = (alsoReports ? [problem, alsoReports] : [problem]).map(
        (text) => `src/content/i18n/de.json: ${text}`
      );
      const problems = checkTranslations(root);
      assert.equal(problems.length, expected.length, problems.join('\n'));
      expected.forEach((text, index) => assert.ok(problems[index].startsWith(text), problems[index]));
    });
  }

  it('flags UI strings that are not valid JSON', () => {
    write('src/content/i18n/de.json', '{ "nav.siteNavLabel": ');
    const [problem, ...rest] = checkTranslations(root);
    assert.deepEqual(rest, []);
    assert.match(problem, /^src\/content\/i18n\/de\.json: not valid JSON/);
  });
});
