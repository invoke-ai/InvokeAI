// Checks on what Crowdin writes into the repository: translated pages under `src/content/docs/<locale>/` and UI strings
// in `src/content/i18n/<locale>.json`. Translated MDX is compiled into the site like any page, Markdown pages pass raw
// HTML through, and Starlight renders the hero and card titles as HTML, so a translation may change words but not
// code, markup or destinations. Pages are parsed as the build parses them and compared with their English source.
// README.md explains what a reviewer still has to read.

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { isDeepStrictEqual } from 'node:util';

import { parseFrontmatter } from '@astrojs/markdown-remark';
import { createProcessor } from '@mdx-js/mdx';

import { translatedLocales } from '../config/locales.mjs';
import { PLACEHOLDER, PLURAL_SUFFIX, parseTags } from './i18n.mjs';

/**
 * A remark plugin from the copy `owner` (a package the build uses) depends on, so these checks parse with the plugins
 * the build parses with.
 * @param {string} name
 * @param {string} owner
 */
const loadBuildPlugin = async (name, owner) => {
  try {
    return (await import(pathToFileURL(createRequire(import.meta.resolve(owner)).resolve(name)).href)).default;
  } catch (error) {
    throw new Error(`translation-checks.mjs parses pages with ${owner}'s ${name}, which it could not load.`, {
      cause: error,
    });
  }
};

// The syntax extensions of the build's Markdown and MDX pipelines, in its order: Astro's GitHub Flavored Markdown
// (autolinked URLs, footnotes, tables), then Starlight's directives (`:::note` asides). The build's other remark
// plugins, ours included, only transform the tree.
const remarkPlugins = [
  await loadBuildPlugin('remark-gfm', '@astrojs/markdown-remark'),
  await loadBuildPlugin('remark-directive', '@astrojs/starlight/schema'),
];
const processors = {
  md: createProcessor({ format: 'md', remarkPlugins }),
  mdx: createProcessor({ format: 'mdx', remarkPlugins }),
};

const DOCS = 'src/content/docs';
const I18N = 'src/content/i18n';

/**
 * The English directories in `src/content/docs/`. Anything else there must be a configured locale, so a language
 * Crowdin exports but `locales.mjs` lacks fails here rather than becoming English pages under its prefix. A new docs
 * tree goes here, and into `crowdin.yml` too if translators should see it.
 */
const ENGLISH_TREES = [
  'assets',
  'configuration',
  'contributing',
  'development',
  'start-here',
  'troubleshooting',
  'Users Guide',
];

/** The page extensions Starlight's docs loader reads. */
const PAGE_EXTENSION = /\.(md|mdx|markdown|mdown|mkdn|mkd|mdwn)$/;

/** Component attributes that hold prose, which translations may change. Starlight renders card titles as HTML. */
const PROSE_ATTRIBUTES = new Set(['alt', 'aria-label', 'description', 'label', 'title']);

/** Frontmatter that holds prose. Every other value is copied from English; the hero renders as HTML. */
const PROSE_FRONTMATTER = /^(title|description|hero\.title|hero\.tagline|sidebar\.label|hero\.actions\.\d+\.text)$/;

/** What UI strings never contain. They render as text, but a tag such as `<link>` wraps a real link. */
const UNSAFE_MESSAGE_PATTERNS = [
  { label: '<script>', pattern: /<script\b/i },
  { label: 'an on… event handler attribute', pattern: /\bon[a-z]+\s*=/i },
  { label: 'a javascript: URL', pattern: /javascript\s*:/i },
];

/**
 * Every problem with the translations under `root`, a docs package directory, as messages that name the file (and
 * the key, element or value) to fix. Empty when the translations are safe to build.
 * @param {string} root
 * @returns {string[]}
 */
export const checkTranslations = (root) => [
  ...checkLocaleDirectories(root),
  ...translatedLocales.flatMap((locale) => checkPages(root, locale)),
  ...translatedLocales.flatMap((locale) => checkMessages(root, locale)),
];

/** @param {string} root */
const checkLocaleDirectories = (root) => {
  const problems = [];
  for (const entry of readdirSync(path.join(root, DOCS), { withFileTypes: true })) {
    if (!entry.isDirectory() || ENGLISH_TREES.includes(entry.name) || translatedLocales.includes(entry.name)) continue;
    problems.push(
      `${DOCS}/${entry.name}/: neither a configured locale nor an English docs tree. If Crowdin exported a language ` +
        `that src/config/locales.mjs lacks, add the locale there (README.md, "Adding a locale") or remove the ` +
        `language from Crowdin. If it is a new English tree, add it to ENGLISH_TREES in src/lib/translation-checks.mjs.`
    );
  }
  const messageFiles = ['en.json', ...translatedLocales.map((locale) => `${locale}.json`)];
  for (const name of readdirSync(path.join(root, I18N))) {
    if (!messageFiles.includes(name)) {
      problems.push(`${I18N}/${name}: not en.json or a configured locale's file (${messageFiles.join(', ')}).`);
    }
  }
  return problems;
};

/**
 * @param {string} root
 * @param {string} locale
 */
const checkPages = (root, locale) => {
  const localeDirectory = path.join(root, DOCS, locale);
  if (!existsSync(localeDirectory)) return [];
  return readdirSync(localeDirectory, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .flatMap((entry) => {
      const filePath = path.join(entry.parentPath, entry.name);
      const relative = path.relative(localeDirectory, filePath).split(path.sep).join('/');
      const file = `${DOCS}/${locale}/${relative}`;
      // A translated page's relative imports and images fall back to the English page's files unless the locale
      // ships its own (plugins/remark-localize-assets.mjs), so any other file here would replace an English one.
      if (!PAGE_EXTENSION.test(entry.name)) return [`${file}: not a page. Crowdin writes only translated pages here.`];
      const sourcePath = path.join(root, DOCS, relative);
      if (!existsSync(sourcePath)) return [`${file}: no English page at ${DOCS}/${relative}.`];
      const format = relative.endsWith('.mdx') ? 'mdx' : 'md';
      return comparePage(readFileSync(sourcePath, 'utf8'), readFileSync(filePath, 'utf8'), format).map(
        (problem) => `${file}: ${problem}`
      );
    });
};

/**
 * What a translated page adds beyond its English source.
 * @param {string} english
 * @param {string} translated
 * @param {'md' | 'mdx'} format
 * @returns {string[]}
 */
const comparePage = (english, translated, format) => {
  let source;
  let page;
  try {
    source = inventory(english, format);
  } catch (error) {
    return [`its English page cannot be parsed: ${firstLine(error)}`];
  }
  try {
    page = inventory(translated, format);
  } catch (error) {
    return [`cannot be parsed: ${firstLine(error)}`];
  }

  const problems = [...frontmatterProblems(source.frontmatter, page.frontmatter, [])];
  for (const value of extra(page.esm, source.esm)) {
    problems.push(`adds the import or export code ${show(value)}. Keep import and export lines as in English.`);
  }
  for (const value of extra(source.esm, page.esm)) {
    problems.push(`lacks the English page's ${show(value)}. Keep import and export lines as in English.`);
  }
  for (const value of extra(page.expressions, source.expressions)) {
    problems.push(`adds the expression ${show(value)}, which the English page does not have.`);
  }
  for (const name of page.elements) {
    if (!source.elements.has(name)) problems.push(`adds ${show(name)}, which the English page does not use.`);
  }
  for (const value of page.attributes) {
    if (!source.attributes.has(value)) problems.push(`adds ${show(value)}, which the English page does not have.`);
  }
  for (const { attribute, value } of page.proseAttributes) {
    if (!source.attributes.has(attribute)) {
      problems.push(`adds ${show(attribute)}, which the English page does not have.`);
    } else if (value.includes('<')) {
      problems.push(`${show(attribute)} contains "<". Attributes take no markup, and card titles render as HTML.`);
    }
  }
  for (const url of page.urls) {
    if (!source.urls.has(url)) problems.push(`links to ${show(url)}, which the English page does not.`);
  }
  for (const html of page.html) {
    if (!source.html.has(html)) problems.push(`adds the HTML ${show(html)}, which the English page does not have.`);
  }
  return [...new Set(problems)];
};

/**
 * The code, markup and destinations of a page, from the syntax tree the build parses: frontmatter, ESM, expressions
 * (comments aside), JSX elements and container directives with their attributes, link and image URLs, and raw HTML.
 * @param {string} text
 * @param {'md' | 'mdx'} format
 */
const inventory = (text, format) => {
  const { frontmatter, content } = parseFrontmatter(text.replace(/\r\n?/g, '\n'), {
    frontmatter: 'empty-with-spaces',
  });
  const result = {
    frontmatter,
    esm: [],
    expressions: [],
    elements: new Set(),
    /** Non-prose attributes, with their values: `<a href="/x/">`. */
    attributes: new Set(),
    /** @type {{ attribute: string, value: string }[]} */
    proseAttributes: [],
    urls: new Set(),
    html: new Set(),
  };
  walk(processors[format].parse(content), (node) => {
    switch (node.type) {
      case 'mdxjsEsm':
        result.esm.push(...node.value.split('\n').flatMap((line) => line.trim() || []));
        break;
      case 'mdxFlowExpression':
      case 'mdxTextExpression':
        // `{/* … */}` compiles to nothing, so translators may translate comments.
        if (node.data?.estree?.body.length !== 0) result.expressions.push(`{${node.value}}`);
        break;
      case 'mdxJsxFlowElement':
      case 'mdxJsxTextElement': {
        const element = node.name ?? '';
        result.elements.add(`<${element}>`);
        for (const attribute of node.attributes) {
          if (attribute.type === 'mdxJsxExpressionAttribute') {
            result.expressions.push(`<${element} {${attribute.value}}>`);
          } else if (typeof attribute.value === 'object' && attribute.value !== null) {
            result.expressions.push(`<${element} ${attribute.name}={${attribute.value.value}}>`);
          } else if (PROSE_ATTRIBUTES.has(attribute.name) && typeof attribute.value === 'string') {
            result.attributes.add(`<${element} ${attribute.name}>`);
            result.proseAttributes.push({ attribute: `<${element} ${attribute.name}>`, value: attribute.value });
          } else {
            const value = attribute.name === 'href' ? withoutFragment(attribute.value) : attribute.value;
            result.attributes.add(`<${element} ${attribute.name}=${JSON.stringify(value)}>`);
          }
        }
        break;
      }
      case 'link':
      case 'image':
      case 'definition':
        result.urls.add(withoutFragment(node.url));
        break;
      case 'html':
        result.html.add(node.value);
        break;
      // Starlight turns `:::note`-style containers into asides and other containers into plain `<div>`s. It restores
      // text and leaf directives, which prose such as "10:30" makes, to plain text, so only containers count.
      case 'containerDirective':
        result.elements.add(`:::${node.name}`);
        for (const [name, value] of Object.entries(node.attributes ?? {})) {
          result.attributes.add(`:::${node.name}{${name}=${JSON.stringify(value)}}`);
        }
        break;
    }
  });
  return result;
};

/** Translations may change a link's fragment, since translated headings get translated ids. */
const withoutFragment = (url) => (typeof url === 'string' ? url.replace(/#.*$/s, '') : url);

/**
 * @param {{ type: string, children?: unknown[] }} node
 * @param {(node: any) => void} visit
 */
const walk = (node, visit) => {
  visit(node);
  for (const child of node.children ?? []) walk(child, visit);
};

/**
 * Where translated frontmatter departs from English: only prose may differ, and it takes no markup.
 * @param {unknown} english
 * @param {unknown} translated
 * @param {string[]} keyPath
 * @returns {string[]}
 */
const frontmatterProblems = (english, translated, keyPath) => {
  const key = keyPath.join('.');
  if (translated !== undefined && english === undefined) {
    return [`frontmatter \`${key}\` is not in the English page.`];
  }
  if (PROSE_FRONTMATTER.test(key)) {
    if (translated === undefined) return [];
    if (typeof translated !== 'string') return [`frontmatter \`${key}\` is not text.`];
    return translated.includes('<') ? [`frontmatter \`${key}\` contains "<". Frontmatter takes no markup.`] : [];
  }
  const isContainer = (value) => typeof value === 'object' && value !== null && !(value instanceof Date);
  if (isContainer(english) && isContainer(translated) && Array.isArray(english) === Array.isArray(translated)) {
    const keys = new Set([...Object.keys(english), ...Object.keys(translated)]);
    return [...keys].flatMap((child) => frontmatterProblems(english[child], translated[child], [...keyPath, child]));
  }
  return isDeepStrictEqual(english, translated)
    ? []
    : [`frontmatter \`${key}\` differs from the English page. Only titles, descriptions and labels are translated.`];
};

/** The items of `list` beyond their count in `allowed`. */
const extra = (list, allowed) => {
  const remaining = new Map();
  for (const item of allowed) remaining.set(item, (remaining.get(item) ?? 0) + 1);
  return list.filter((item) => {
    const count = remaining.get(item) ?? 0;
    remaining.set(item, count - 1);
    return count <= 0;
  });
};

/** @param {string} value */
const show = (value) => `\`${value.length > 100 ? `${value.slice(0, 99)}…` : value}\``;

/** @param {unknown} error */
const firstLine = (error) => String(error instanceof Error ? error.message : error).split('\n')[0];

/**
 * @param {string} root
 * @param {string} locale
 */
const checkMessages = (root, locale) => {
  const file = `${I18N}/${locale}.json`;
  if (!existsSync(path.join(root, file))) return [];
  const english = JSON.parse(readFileSync(path.join(root, I18N, 'en.json'), 'utf8'));
  let messages;
  try {
    messages = JSON.parse(readFileSync(path.join(root, file), 'utf8'));
  } catch (error) {
    return [`${file}: not valid JSON (${firstLine(error)}).`];
  }
  if (typeof messages !== 'object' || messages === null || Array.isArray(messages)) {
    return [`${file}: not a JSON object of messages.`];
  }
  return Object.entries(messages).flatMap(([key, message]) =>
    compareMessage(english, key, message).map((problem) => `${file}: "${key}" ${problem}`)
  );
};

/**
 * What is wrong with one translated message, given all of `en.json`.
 * @param {Record<string, string>} english
 * @param {string} key
 * @param {unknown} message
 * @returns {string[]}
 */
const compareMessage = (english, key, message) => {
  if (typeof message !== 'string') return ['is not a string.'];

  // A plural form may be one English lacks (`_few`, `_many`), and may leave out `{{count}}` or a tag, as in "one
  // result"; it may use only what some English form of the message does.
  const base = key.replace(PLURAL_SUFFIX, '');
  const pluralSources = PLURAL_SUFFIX.test(key)
    ? Object.keys(english)
        .filter((englishKey) => PLURAL_SUFFIX.test(englishKey) && englishKey.replace(PLURAL_SUFFIX, '') === base)
        .map((englishKey) => english[englishKey])
    : [];
  const isPluralForm = pluralSources.length > 0;
  const sources = isPluralForm ? pluralSources : Object.hasOwn(english, key) ? [english[key]] : [];
  if (sources.length === 0) return ['is not in en.json. Remove it, or add the English string first.'];

  const problems = [];
  for (const { label, pattern } of UNSAFE_MESSAGE_PATTERNS) {
    if (pattern.test(message)) problems.push(`contains ${label}.`);
  }

  const compare = (noun, translatedNames, englishNames) => {
    const added = [...translatedNames].filter((name) => !englishNames.has(name));
    const missing = isPluralForm ? [] : [...englishNames].filter((name) => !translatedNames.has(name));
    if (added.length > 0) problems.push(`has ${noun} English does not: ${added.join(', ')}.`);
    if (missing.length > 0) problems.push(`is missing ${noun}: ${missing.join(', ')}.`);
  };
  const placeholders = (text) => new Set([...text.matchAll(PLACEHOLDER)].map((match) => `{{${match[1]}}}`));
  const tags = (text) => new Set(parseTags(text).flatMap((part) => ('tag' in part ? [`<${part.tag}>`] : [])));
  compare('placeholders', placeholders(message), new Set(sources.flatMap((text) => [...placeholders(text)])));
  compare('tags', tags(message), new Set(sources.flatMap((text) => [...tags(text)])));
  return problems;
};
