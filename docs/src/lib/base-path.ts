export const withBase = (path: string, baseUrl: string) => {
  const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const normalizedPath = path.replace(/^\//, '');

  return `${normalizedBase}${normalizedPath}`;
};
