export interface FontListParams {
  contentHash?: string;
  offset?: number;
  limit?: number;
  search?: string;
  scope?: 'private' | 'shared' | 'all';
}

/** Query keys are kept independent from the transport so account-owned callers
 * can invalidate the catalog without importing the font byte/API module. */
export const fontKeys = {
  all: ['fonts'] as const,
  catalog: (params: FontListParams = {}) => ['fonts', 'catalog', params] as const,
  infiniteCatalog: (params: FontListParams = {}) => ['fonts', 'infinite-catalog', params] as const,
  detail: (id: string) => ['fonts', 'detail', id] as const,
};
