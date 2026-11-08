/**
 * A new-issue link for a documentation page, using the `DOCS_ISSUE.yml` issue form with the page URL filled in.
 * @param title the page title
 * @param pageUrl the page's absolute URL
 */
export const getReportIssueUrl = (title: string, pageUrl: string) => {
  const url = new URL('https://github.com/invoke-ai/InvokeAI/issues/new');
  url.searchParams.set('template', 'DOCS_ISSUE.yml');
  url.searchParams.set('title', `[docs]: ${title}`);
  url.searchParams.set('page', pageUrl);
  return url.href;
};
