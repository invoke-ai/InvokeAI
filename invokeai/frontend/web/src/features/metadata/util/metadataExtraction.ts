/**
 * Utility functions for extracting metadata from different sources
 */

/**
 * Type guard to check if an object has a session property
 */
const hasSession = (data: unknown): data is { session: { graph?: { nodes?: Record<string, unknown> } } } => {
  return (
    data !== null &&
    typeof data === 'object' &&
    'session' in data &&
    typeof (data as Record<string, unknown>).session === 'object' &&
    (data as Record<string, unknown>).session !== null
  );
};

/**
 * Type guard to check if an object has a metadata property
 */
const hasMetadata = (data: unknown): data is { metadata: unknown } => {
  return data !== null && typeof data === 'object' && 'metadata' in data;
};

/**
 * Extracts metadata from a session graph
 * @param session The session object containing the graph
 * @returns The extracted metadata or null if not found
 */
export const extractMetadataFromSession = (
  session: { graph?: { nodes?: Record<string, unknown> } } | null
): unknown => {
  if (!session?.graph?.nodes) {
    return null;
  }

  // Find the metadata node (core_metadata with unique suffix)
  const nodeKeys = Object.keys(session.graph.nodes);
  const metadataNodeKey = nodeKeys.find((key) => key.startsWith('core_metadata:'));

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
  if (hasSession(data)) {
    const sessionMetadata = extractMetadataFromSession(data.session);
    if (sessionMetadata) {
      return sessionMetadata;
    }
  }

  // Try to extract from image DTO
  if (hasMetadata(data)) {
    const imageMetadata = extractMetadataFromImage(data);
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
