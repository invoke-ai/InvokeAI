import { useMemo } from 'react';
import { extractMetadata } from 'features/metadata/util/metadataExtraction';

/**
 * Hook for extracting metadata from different data structures
 * @param data The data object that might contain metadata
 * @returns The extracted metadata or null if not found
 */
export const useMetadataExtraction = (data: unknown): unknown => {
  return useMemo(() => {
    return extractMetadata(data);
  }, [data]);
}; 