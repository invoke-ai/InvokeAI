import { useMemo } from 'react';
import { useGetRelatedModelIdsBatchQuery } from 'services/api/endpoints/modelRelationships';

/**
 * Fetches related model keys for a given set of selected model keys.
 * Returns a Set<string> for fast lookup.
 */
export const useRelatedModelKeys = (selectedKeys: Set<string>) => {
  const { data: related = [] } = useGetRelatedModelIdsBatchQuery([...selectedKeys], {
    skip: selectedKeys.size === 0,
  });

  return useMemo(() => new Set(related), [related]);
};
