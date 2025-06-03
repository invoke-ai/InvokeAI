import { EMPTY_ARRAY } from 'app/store/constants';
import { useMemo } from 'react';
import { useGetRelatedModelIdsBatchQuery } from 'services/api/endpoints/modelRelationships';

const options: Parameters<typeof useGetRelatedModelIdsBatchQuery>[1] = {
  selectFromResult: ({ data }) => {
    if (!data) {
      return { related: EMPTY_ARRAY };
    }
    return data;
  },
};

/**
 * Fetches related model keys for a given set of selected model keys.
 * Returns a Set<string> for fast lookup.
 */
export const useRelatedModelKeys = (selectedKeys: string[]) => {
  const { related } = useGetRelatedModelIdsBatchQuery(selectedKeys, options);

  return useMemo(() => new Set(related), [related]);
};
