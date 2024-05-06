import { forEach, groupBy, isEqual, values } from 'lodash-es';
import type {
  AnyInvocation,
  AnyInvocationInputField,
  AnyInvocationOutputField,
  InputFields,
  Invocation,
  InvocationType,
  OutputFields,
} from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

type Edge = {
  source: {
    node_id: string;
    field: AnyInvocationOutputField;
  };
  destination: {
    node_id: string;
    field: AnyInvocationInputField;
  };
};

export type GraphType = { id: string; nodes: Record<string, AnyInvocation>; edges: Edge[] };

export class Graph {
  _graph: GraphType;

  constructor(id?: string) {
    this._graph = {
      id: id ?? Graph.uuid(),
      nodes: {},
      edges: [],
    };
  }

  //#region Node Operations

  /**
   * Add a node to the graph. If a node with the same id already exists, an `AssertionError` is raised.
   * The optional `is_intermediate` and `use_cache` fields are set to `true` and `true` respectively if not set on the node.
   * @param node The node to add.
   * @returns The added node.
   * @raises `AssertionError` if a node with the same id already exists.
   */
  addNode<T extends InvocationType>(node: Invocation<T>): Invocation<T> {
    assert(this._graph.nodes[node.id] === undefined, Graph.getNodeAlreadyExistsMsg(node.id));
    if (node.is_intermediate === undefined) {
      node.is_intermediate = true;
    }
    if (node.use_cache === undefined) {
      node.use_cache = true;
    }
    this._graph.nodes[node.id] = node;
    return node;
  }

  /**
   * Gets a node from the graph.
   * @param id The id of the node to get.
   * @returns The node.
   * @raises `AssertionError` if the node does not exist or if a `type` is provided but the node is not of the expected type.
   */
  getNode(id: string): AnyInvocation {
    const node = this._graph.nodes[id];
    assert(node !== undefined, Graph.getNodeNotFoundMsg(id));
    return node;
  }

  /**
   * Check if a node exists in the graph.
   * @param id The id of the node to check.
   */
  hasNode(id: string): boolean {
    try {
      this.getNode(id);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get the immediate incomers of a node.
   * @param nodeId The id of the node to get the incomers of.
   * @returns The incoming nodes.
   * @raises `AssertionError` if the node does not exist.
   */
  getIncomers(node: AnyInvocation): AnyInvocation[] {
    return this.getEdgesTo(node).map((edge) => this.getNode(edge.source.node_id));
  }

  /**
   * Get the immediate outgoers of a node.
   * @param nodeId The id of the node to get the outgoers of.
   * @returns The outgoing nodes.
   * @raises `AssertionError` if the node does not exist.
   */
  getOutgoers(node: AnyInvocation): AnyInvocation[] {
    return this.getEdgesFrom(node).map((edge) => this.getNode(edge.destination.node_id));
  }
  //#endregion

  //#region Edge Operations

  /**
   * Add an edge to the graph. If an edge with the same source and destination already exists, an `AssertionError` is raised.
   * If providing node ids, provide the from and to node types as generics to get type hints for from and to field names.
   * @param fromNode The source node or id of the source node.
   * @param fromField The field of the source node.
   * @param toNode The source node or id of the destination node.
   * @param toField The field of the destination node.
   * @returns The added edge.
   * @raises `AssertionError` if an edge with the same source and destination already exists.
   */
  addEdge<TFrom extends AnyInvocation, TTo extends AnyInvocation>(
    fromNode: TFrom,
    fromField: OutputFields<TFrom>,
    toNode: TTo,
    toField: InputFields<TTo>
  ): Edge {
    const edge: Edge = {
      source: { node_id: fromNode.id, field: fromField },
      destination: { node_id: toNode.id, field: toField },
    };
    const edgeAlreadyExists = this._graph.edges.some((e) => isEqual(e, edge));
    assert(!edgeAlreadyExists, Graph.getEdgeAlreadyExistsMsg(fromNode.id, fromField, toNode.id, toField));
    this._graph.edges.push(edge);
    return edge;
  }

  /**
   * Add an edge to the graph. If an edge with the same source and destination already exists, an `AssertionError` is raised.
   * If providing node ids, provide the from and to node types as generics to get type hints for from and to field names.
   * @param fromNode The source node or id of the source node.
   * @param fromField The field of the source node.
   * @param toNode The source node or id of the destination node.
   * @param toField The field of the destination node.
   * @returns The added edge.
   * @raises `AssertionError` if an edge with the same source and destination already exists.
   */
  addEdgeFromObj(edge: Edge): Edge {
    const edgeAlreadyExists = this._graph.edges.some((e) => isEqual(e, edge));
    assert(
      !edgeAlreadyExists,
      Graph.getEdgeAlreadyExistsMsg(
        edge.source.node_id,
        edge.source.field,
        edge.destination.node_id,
        edge.destination.field
      )
    );
    this._graph.edges.push(edge);
    return edge;
  }

  /**
   * Get an edge from the graph. If the edge does not exist, an `AssertionError` is raised.
   * Provide the from and to node types as generics to get type hints for from and to field names.
   * @param fromNodeId The id of the source node.
   * @param fromField The field of the source node.
   * @param toNodeId The id of the destination node.
   * @param toField The field of the destination node.
   * @returns The edge.
   * @raises `AssertionError` if the edge does not exist.
   */
  getEdge<TFrom extends AnyInvocation, TTo extends AnyInvocation>(
    fromNode: TFrom,
    fromField: OutputFields<TFrom>,
    toNode: TTo,
    toField: InputFields<TTo>
  ): Edge {
    const edge = this._graph.edges.find(
      (e) =>
        e.source.node_id === fromNode.id &&
        e.source.field === fromField &&
        e.destination.node_id === toNode.id &&
        e.destination.field === toField
    );
    assert(edge !== undefined, Graph.getEdgeNotFoundMsg(fromNode.id, fromField, toNode.id, toField));
    return edge;
  }

  /**
   * Check if a graph has an edge.
   * Provide the from and to node types as generics to get type hints for from and to field names.
   * @param fromNodeId The id of the source node.
   * @param fromField The field of the source node.
   * @param toNodeId The id of the destination node.
   * @param toField The field of the destination node.
   * @returns Whether the graph has the edge.
   */

  hasEdge<TFrom extends AnyInvocation, TTo extends AnyInvocation>(
    fromNode: TFrom,
    fromField: OutputFields<TFrom>,
    toNode: TTo,
    toField: InputFields<TTo>
  ): boolean {
    try {
      this.getEdge(fromNode, fromField, toNode, toField);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get all edges from a node. If `fromField` is provided, only edges from that field are returned.
   * Provide the from node type as a generic to get type hints for from field names.
   * @param fromNodeId The id of the source node.
   * @param fromField The field of the source node (optional).
   * @returns The edges.
   */
  getEdgesFrom<T extends AnyInvocation>(fromNode: T, fromField?: OutputFields<T>): Edge[] {
    let edges = this._graph.edges.filter((edge) => edge.source.node_id === fromNode.id);
    if (fromField) {
      edges = edges.filter((edge) => edge.source.field === fromField);
    }
    return edges;
  }

  /**
   * Get all edges to a node. If `toField` is provided, only edges to that field are returned.
   * Provide the to node type as a generic to get type hints for to field names.
   * @param toNodeId The id of the destination node.
   * @param toField The field of the destination node (optional).
   * @returns The edges.
   */
  getEdgesTo<T extends AnyInvocation>(toNode: T, toField?: InputFields<T>): Edge[] {
    let edges = this._graph.edges.filter((edge) => edge.destination.node_id === toNode.id);
    if (toField) {
      edges = edges.filter((edge) => edge.destination.field === toField);
    }
    return edges;
  }

  /**
   * Delete _all_ matching edges from the graph. Uses _.isEqual for comparison.
   * @param edge The edge to delete
   */
  private _deleteEdge(edge: Edge): void {
    this._graph.edges = this._graph.edges.filter((e) => !isEqual(e, edge));
  }

  /**
   * Delete all edges to a node. If `toField` is provided, only edges to that field are deleted.
   * Provide the to node type as a generic to get type hints for to field names.
   * @param toNode The destination node.
   * @param toField The field of the destination node (optional).
   */
  deleteEdgesTo<T extends AnyInvocation>(toNode: T, toField?: InputFields<T>): void {
    for (const edge of this.getEdgesTo(toNode, toField)) {
      this._deleteEdge(edge);
    }
  }

  /**
   * Delete all edges from a node. If `fromField` is provided, only edges from that field are deleted.
   * Provide the from node type as a generic to get type hints for from field names.
   * @param toNodeId The id of the source node.
   * @param toField The field of the source node (optional).
   */
  deleteEdgesFrom<T extends AnyInvocation>(fromNode: T, fromField?: OutputFields<T>): void {
    for (const edge of this.getEdgesFrom(fromNode, fromField)) {
      this._deleteEdge(edge);
    }
  }
  //#endregion

  //#region Graph Ops

  /**
   * Validate the graph. Checks that all edges have valid source and destination nodes.
   * TODO(psyche): Add more validation checks - cycles, valid invocation types, etc.
   * @raises `AssertionError` if an edge has an invalid source or destination node.
   */
  validate(): void {
    for (const edge of this._graph.edges) {
      this.getNode(edge.source.node_id);
      this.getNode(edge.destination.node_id);
      assert(
        !this._graph.edges.filter((e) => e !== edge).find((e) => isEqual(e, edge)),
        `Duplicate edge: ${Graph.edgeToString(edge)}`
      );
    }
    for (const node of values(this._graph.nodes)) {
      const edgesTo = this.getEdgesTo(node);
      // Validate that no node has multiple incoming edges with the same field
      forEach(groupBy(edgesTo, 'destination.field'), (group, field) => {
        if (node.type === 'collect' && field === 'item') {
          // Collectors' item field accepts multiple incoming edges
          return;
        }
        assert(
          group.length === 1,
          `Node ${node.id} has multiple incoming edges with field ${field}: ${group.map(Graph.edgeToString).join(', ')}`
        );
      });
    }
  }

  /**
   * Gets the graph after validating it.
   * @returns The graph.
   * @raises `AssertionError` if the graph is invalid.
   */
  getGraph(): GraphType {
    this.validate();
    return this._graph;
  }

  /**
   * Gets the graph without validating it.
   * @returns The graph.
   */
  getGraphSafe(): GraphType {
    return this._graph;
  }
  //#endregion

  //#region Util

  static getNodeNotFoundMsg(id: string): string {
    return `Node ${id} not found`;
  }

  static getNodeNotOfTypeMsg(node: AnyInvocation, expectedType: InvocationType): string {
    return `Node ${node.id} is not of type ${expectedType}: ${node.type}`;
  }

  static getNodeAlreadyExistsMsg(id: string): string {
    return `Node ${id} already exists`;
  }

  static getEdgeNotFoundMsg(fromNodeId: string, fromField: string, toNodeId: string, toField: string) {
    return `Edge from ${fromNodeId}.${fromField} to ${toNodeId}.${toField} not found`;
  }

  static getEdgeAlreadyExistsMsg(fromNodeId: string, fromField: string, toNodeId: string, toField: string) {
    return `Edge from ${fromNodeId}.${fromField} to ${toNodeId}.${toField} already exists`;
  }

  static edgeToString(edge: Edge): string {
    return `${edge.source.node_id}.${edge.source.field} -> ${edge.destination.node_id}.${edge.destination.field}`;
  }

  static uuid = uuidv4;
  //#endregion
}
