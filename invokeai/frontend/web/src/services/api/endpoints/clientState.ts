import { api, buildV1Url, LIST_TAG } from '..';

/**
 * Builds an endpoint URL for the client_state router.
 * The queue_id path parameter is kept as 'default' for backwards compatibility.
 */
const buildClientStateUrl = (path: string, query?: Record<string, string>) =>
  buildV1Url(`client_state/default/${path}`, query);

export const clientStateApi = api.injectEndpoints({
  endpoints: (build) => ({
    getClientStateKeysByPrefix: build.query<string[], string>({
      query: (prefix) => ({
        url: buildClientStateUrl('get_keys_by_prefix', { prefix }),
        method: 'GET',
      }),
      providesTags: [{ type: 'ClientState', id: LIST_TAG }, 'FetchOnReconnect'],
    }),
    getClientStateByKey: build.query<string | null, string>({
      query: (key) => ({
        url: buildClientStateUrl('get_by_key', { key }),
        method: 'GET',
      }),
    }),
    setClientStateByKey: build.mutation<string, { key: string; value: string }>({
      query: ({ key, value }) => ({
        url: buildClientStateUrl('set_by_key', { key }),
        method: 'POST',
        // Send raw string body — the backend expects Body(...) as a plain string,
        // not JSON-encoded. Setting Content-Type to text/plain prevents fetchBaseQuery
        // from JSON-stringifying the body.
        headers: { 'Content-Type': 'text/plain' },
        body: value,
      }),
      invalidatesTags: [{ type: 'ClientState', id: LIST_TAG }],
    }),
    deleteClientStateByKey: build.mutation<void, string>({
      query: (key) => ({
        url: buildClientStateUrl('delete_by_key', { key }),
        method: 'POST',
      }),
      invalidatesTags: [{ type: 'ClientState', id: LIST_TAG }],
    }),
  }),
});

export const { useGetClientStateKeysByPrefixQuery, useSetClientStateByKeyMutation, useDeleteClientStateByKeyMutation } =
  clientStateApi;
