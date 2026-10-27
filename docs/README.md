# Invoke AI Documentation

The docs site at [v7.invoke.ai](https://v7.invoke.ai), built with Astro Starlight. Paths below are relative to `docs/`; `crowdin.yml` is at the repository root.

Contributor guide: [Documentation](src/content/docs/development/Documentation/index.mdx) (dev server, adding pages, builds, tests).

Toolchain: Node 22.23.4 or newer (`engines` in `package.json`; Astro needs it) and pnpm 10, run from this directory.

## Translations

English is the source of truth. Translators work in the Crowdin project ([crowdin.com/project/invoke](https://crowdin.com/project/invoke), `CROWDIN_PROJECT_URL` in `src/config/locales.mjs`; confirm this URL once the project is set up, since `src/content/docs/contributing/translations.md` links it too). Crowdin's GitHub integration reads `crowdin.yml`, uploads the English sources from `main` and opens a pull request with the translations. There is no Crowdin workflow or repository secret.

The site builds the locales in `src/config/locales.mjs`: English at the root, and `de`, `es` and `hi` under `/<locale>/`. A page without a translation is served in English with Starlight's "not translated yet" notice. A translated page's footer link reads **Translate this page** and opens Crowdin instead of GitHub.

### What Crowdin owns

- `src/content/docs/{de,es,hi}/**`: translated pages at the English page's path.
- `src/content/i18n/{de,es,hi}.json`: translated interface strings.

Don't edit these by hand: the next sync overwrites them. Fix the English source, or the translation in Crowdin.

Crowdin gets the home page (`src/content/docs/index.mdx`), `start-here/`, `Users Guide/`, `configuration/`, `troubleshooting/` and `src/content/i18n/en.json`. `development/` and `contributing/` stay English and show as fallbacks in every locale. Releases, download and showcase pages exist only at the root, in English.

Translations are not marked `linguist-generated` in `.gitattributes`: translated MDX compiles into the site, so its diffs stay expanded for review.

### Adding pages and UI strings

- **Pages.** A new `.mdx` page in a translated tree reaches Crowdin on the next sync. A tree's first `.md` page needs an `.md` entry in `crowdin.yml` (the Crowdin CLI fails on patterns that match nothing, so only `Users Guide/` has one). A new top-level docs directory goes into `ENGLISH_TREES` in `src/lib/translation-checks.mjs`, and into `crowdin.yml` if it should be translated.
- **UI strings.** Add the English string to `src/content/i18n/en.json` and read it with `Astro.locals.t('key')`, or with the `Trans` component (`src/lib/components/Trans.astro`) when it contains markup such as `<link>…</link>`. Placeholders are i18next `{{name}}`. Plurals are `key_one`/`key_other`, read with `t('key', { count })`, or in a client script with `pluralForms` (`src/lib/i18n.mjs`). `src/lib/i18n-keys.test.mjs` fails when code uses a key `en.json` lacks, or `en.json` has a key nothing uses. Give short or ambiguous strings a context note in Crowdin (below).

### Adding a locale

1. Add the language as a target in Crowdin.
2. Add a lowercase key to `locales` in `src/config/locales.mjs`, with the language's own name as `label` and its BCP 47 tag as `lang` (for example `'zh-cn': { label: '简体中文', lang: 'zh-CN' }`).
3. For a regional variant, map Crowdin's code to that lowercase directory, because `%two_letters_code%` drops the region. Add this to every `files` entry in `crowdin.yml`:

   ```yaml
   languages_mapping:
     two_letters_code:
       zh-CN: zh-cn
   ```

Nothing else changes: Starlight's routes, the language menu, `scripts/verify-deploy-output.mjs` and the translation checks all read `locales.mjs`. Check in a build that the site font covers the script (Inter has no Devanagari, for example).

### Reviewing a Crowdin pull request

Docs CI runs `pnpm test`, the build (links included) and the deploy checks on it. `src/lib/translation-checks.mjs` parses each translated page as the build does (Astro's frontmatter parser, then MDX or Markdown with the build's GitHub Flavored Markdown and Starlight directive plugins) and compares it with the English page at the same path. A translation may change prose, comments and code samples; anything else fails with the file and the offending key, element or value:

- a directory in `src/content/docs/` or a file in `src/content/i18n/` for a locale `locales.mjs` doesn't configure;
- a file in a locale directory that isn't a page, or a page with no English page at the same path;
- a page that doesn't parse;
- `import`/`export` code that differs from English;
- a `{…}` expression, JSX element, aside or other container directive (`:::note`), attribute or attribute value the English page lacks. Prose attributes (`alt`, `aria-label`, `description`, `label`, `title`) may change but take no `<`, since Starlight renders card titles as HTML;
- a link, image or reference URL the English page lacks, autolinked URLs included (the `#fragment` may change);
- raw HTML the English page lacks (in `.md` pages);
- frontmatter beyond `title`, `description`, `hero.title`, `hero.tagline`, `sidebar.label` and `hero.actions[].text` that differs from English, such as `head`, `template` or a link, or any of those containing `<`;
- a string in `<locale>.json` with a key `en.json` lacks (plural messages may use any of `_zero`, `_one`, `_two`, `_few`, `_many`, `_other`), different `{{placeholders}}` or tags, or `<script`, `on…=` or `javascript:`.

Fix failures in Crowdin, not in the pull request. The checks don't judge language, so still read for meaning, tone and product names left in English (Invoke, node and model names), for edited commands and values in code samples, and for the rendering in a preview build (`pnpm build && pnpm preview`), especially Hindi.

Before merging the first real translation, run `pnpm test` on the first sync's pull request, or locally on `pnpm dlx @crowdin/cli download --pseudo` output (run from the repository root; it writes into the locale directories, so delete them afterwards). It confirms that Crowdin's MDX v2 export leaves imports, components and expressions byte-identical, which every later check relies on.

### Crowdin parser settings

`crowdin.yml` sets the file types: MDX v2 (`mdx_v2`) for `.mdx`, Markdown for `.md` and i18next JSON for `en.json`, so `_one`/`_other` keys become plural forms. In Crowdin's Settings → Parser configuration, exclude the frontmatter that isn't prose, for both MDX and Markdown:

`lastUpdated`, `template`, `slug`, `sidebar.order`, `sidebar.hidden`, `pagefind`, `draft`, `alsoListedIn`, `hero.actions.link`, `hero.image.file`

### Translator context notes

After the first sync, paste these into each string's context in Crowdin.

| Key | Note |
| --- | --- |
| `search.scopeChipsIntro` | "In", shown before the search scope chips. |
| `search.scopeChangeStatus` | Screen-reader announcement: `{{scope}}` is the scope's name, `{{results}}` the result count message. |
| `search.settingDefault` | Pill on a settings result: "default" and the setting's default value. |
| `search.settingsGroup` | Heading of settings results. Keep `invokeai.yaml`. |
| `keyboardHint.*` | One short verb each, shown after a key symbol. |
| `languageMenu.accessibleLabel` | `{{label}}` is "Select language", `{{code}}` the locale code, `{{language}}` the language's name. |
| `settingsReference.envVar` | "Env", short for environment variable. |
| `settingsReference.allowedValues` | A list of allowed values follows. |
| `topics.group.*` | One-word imperative group headings. |
| `pageActions.copyDone`, `pageActions.copyFailed` | Short button states. |
| `page.updated`, `setupPicker.summary`, `setupPicker.summaryNotes`, `footer.credit`, `pageFeedback.reportPrompt` | Keep the `<time>`, `<setup>` and `<link>` tags around the matching words. |
| `nav.siteNavLabel` | Landmark name for screen readers. |
| `sidebar.group.*` | Navigation section names. |
| `home.hero.announcement` | Keep "Invoke V7". |
| `home.*.eyebrow` | Short label above a home page section heading. |
| `home.directory.*.kicker` | One-word verbs. |
| `home.cta.platformsAndLicense` | Keep the OS names and "Apache 2.0". |
| `home.install.body` | Keep the three tags; their order may change. |
| `home.community.body` | Keep the `<link>` tag and the maintainers' names. |
| `whatsNew.*.body` | Keep node and model names. |
| `modelFamilies.licenseTag`, `modelFamilies.vramTag` | `<srOnly>` text is read only by screen readers. Keep "GB+". |
| `modelFamilies.format.singleFile` | A model file format, listed beside GGUF and int8: one checkpoint file. |
| `imageCompare.valueText` | A slider's spoken value: `{{before}}` and `{{after}}` are percentages, `{{beforeLabel}}` and `{{afterLabel}}` the two images' labels, such as "Before" and "After". |
| `workflowDemo.field` | `{{field}}` and `{{node}}` stay English mock names. |
| `workflowDemo.connected`, `workflowDemo.disconnected` | `{{source}}` and `{{target}}` are filled with the `workflowDemo.field` message. |
| `communityNodes.manualInstallNote` | Keep the `<code>` tags; `nodes` and `__init__.py` stay as they are. |
| `communityNodes.shown_*` | `{{shown}}` packs match the filters, of `{{count}}` in all; the plural form follows `{{count}}`, the total. |
| `communityNodes.copyDone`, `communityNodes.copyFailed` | Short button states. |
| `communityNodes.byAuthor` | `{{author}}` is a GitHub or GitLab user name. |
| `communityNodes.dependencies`, `communityNodes.manualInstall` | Short badge text on a node pack card. |
