import { localizePath } from '../config/locales.mjs';

export const withBase = (path: string, baseUrl: string) => {
  const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const normalizedPath = path.replace(/^\//, '');

  return `${normalizedBase}${normalizedPath}`;
};

/** A site path, without the base path, as a link from a page in `locale`: root-only paths stay at the root. */
export const localizedHref = (path: string, { locale, base }: { locale: string | undefined; base: string }) =>
  withBase(localizePath(path, locale), base);
