// @ts-check

/**
 * The page's language, for number formatting. Routes outside Starlight's pages, such as `llms-full.txt`, also render
 * page content but have no `starlightRoute`, whose getter throws there; they are English.
 * @param {App.Locals} locals
 */
export function pageLang(locals) {
  try {
    return locals.starlightRoute.lang;
  } catch {
    return 'en';
  }
}
