import type { Connection, Edge } from 'reactflow';
import { assert } from 'tsafe';

/**
 * Gets the edge id for a connection
 * Copied from: https://github.com/xyflow/xyflow/blob/v11/packages/core/src/utils/graph.ts#L44-L45
 * Requested for this to be exported in: https://github.com/xyflow/xyflow/issues/4290
 * @param connection The connection to get the id for
 * @returns The edge id
 */
const getEdgeId = (connection: Connection): string => {
  const { source, sourceHandle, target, targetHandle } = connection;
  return `reactflow__edge-${source}${sourceHandle || ''}-${target}${targetHandle || ''}`;
};

/**
 * Converts a connection to an edge
 * @param connection The connection to convert to an edge
 * @returns The edge
 * @throws If the connection is invalid (e.g. missing source, sourcehandle, target, or targetHandle)
 */
export const connectionToEdge = (connection: Connection): Edge => {
  const { source, sourceHandle, target, targetHandle } = connection;
  assert(source && sourceHandle && target && targetHandle, 'Invalid connection');
  return {
    source,
    sourceHandle,
    target,
    targetHandle,
    id: getEdgeId({ source, sourceHandle, target, targetHandle }),
  };
};
