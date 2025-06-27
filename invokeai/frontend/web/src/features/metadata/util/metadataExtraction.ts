/**
 * Utility functions for extracting metadata from different sources
 */

/**
 * Extracts metadata from a session graph
 * @param session The session object containing the graph
 * @returns The extracted metadata or null if not found
 */
export const extractMetadataFromSession = (session: { graph?: { nodes?: Record<string, unknown> } } | null): unknown => {
  if (!session?.graph?.nodes) {
    return null;
  }
  
  // Find the metadata node (core_metadata with unique suffix)
  const nodeKeys = Object.keys(session.graph.nodes);
  const metadataNodeKey = nodeKeys.find(key => key.startsWith('core_metadata:'));
  
  if (!metadataNodeKey) {
    return null;
  }
  
  return session.graph.nodes[metadataNodeKey];
};

/**
 * Extracts metadata from an image DTO
 * @param image The image DTO object
 * @returns The extracted metadata or null if not found
 */
export const extractMetadataFromImage = (image: { metadata?: unknown } | null): unknown => {
  return image?.metadata || null;
};

/**
 * Generic metadata extraction that works with different data structures
 * @param data The data object that might contain metadata
 * @returns The extracted metadata or null if not found
 */
export const extractMetadata = (data: unknown): unknown => {
  if (!data || typeof data !== 'object') {
    return null;
  }

  // Try to extract from session graph
  if ('session' in data && data.session && typeof data.session === 'object') {
    const sessionMetadata = extractMetadataFromSession(data.session as any);
    if (sessionMetadata) {
      return sessionMetadata;
    }
  }

  // Try to extract from image DTO
  if ('metadata' in data) {
    const imageMetadata = extractMetadataFromImage(data as any);
    if (imageMetadata) {
      return imageMetadata;
    }
  }

  // If the data itself looks like metadata, return it
  if (data && typeof data === 'object' && Object.keys(data).length > 0) {
    return data;
  }

  return null;
}; 