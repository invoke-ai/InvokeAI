export const withBase = (path: string, baseUrl: string) => {
  const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const normalizedPath = path.replace(/^\//, '');

  return `${normalizedBase}${normalizedPath}`;
};

export const localizePath = (path: string, locale?: string) => {
  if (!locale) {
    return path;
  }

  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `/${locale}${normalizedPath}`;
};
