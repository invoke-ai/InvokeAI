---
title: Translations
---

InvokeAI uses two translation services:

- [Crowdin](https://crowdin.com/project/invokeai-docs) for this documentation site.
- [Weblate](https://hosted.weblate.org/engage/invokeai/) for text in the InvokeAI application.

The initial documentation languages are German, Spanish, and Hindi. Pages that are not translated yet display the English content with an untranslated-content notice. Partially translated pages may retain some English text.

## Translating the documentation

Visit the [InvokeAI Docs project on Crowdin](https://crowdin.com/project/invokeai-docs), sign in, and select German, Spanish, or Hindi. Crowdin preserves Markdown structure, code examples, links, and other non-translatable syntax while presenting the page text for translation.

English documentation is maintained in GitHub and synchronized to Crowdin. Completed translations are returned in an automated pull request, checked by the normal documentation build, and merged by an InvokeAI maintainer. Do not edit files inside the `docs/src/content/docs/de`, `es`, or `hi` directories directly because Crowdin owns those generated files.

## Translating the application

Visit the [InvokeAI project on Weblate](https://hosted.weblate.org/engage/invokeai/), sign in, select a language, and choose the Web UI component. These translations affect the application interface rather than this website.

## Help & Questions

For documentation translation questions, see [Crowdin's translator guide](https://support.crowdin.com/online-editor/) or ask in the [InvokeAI Discord](https://discord.gg/ZmtBAhwWhy). For application translation questions, see [Weblate's documentation](https://docs.weblate.org/en/latest/index.html).

## Thanks

Thanks to the InvokeAI community for helping make the application and its documentation accessible worldwide!
