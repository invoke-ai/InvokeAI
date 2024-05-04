import { isEqual } from 'lodash-es';
import type {
  AnyInvocation,
  AnyInvocationInputField,
  AnyInvocationOutputField,
  Invocation,
  InvocationInputFields,
  InvocationOutputFields,
  InvocationType,
  S,
} from 'services/api/types';
import type { O } from 'ts-toolbelt';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

type GraphType = O.NonNullable<O.Required<S['Graph']>>;
type Edge = GraphType['edges'][number];
type Never = Record<string, never>;

// The `core_metadata` node has very lax types, it accepts arbitrary field names. It must be excluded from edge utils
// to preview their types from being widened from a union of valid field names to `string | number | symbol`.
type EdgeNodeType = Exclude<InvocationType, 'core_metadata'>;

type EdgeFromField<TFrom extends EdgeNodeType | Never = Never> = TFrom extends EdgeNodeType
  ? InvocationOutputFields<TFrom>
  : AnyInvocationOutputField;

type EdgeToField<TTo extends EdgeNodeType | Never = Never> = TTo extends EdgeNodeType
  ? InvocationInputFields<TTo>
  : AnyInvocationInputField;

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
   * @param type The type of the node to get. If provided, the retrieved node is guaranteed to be of this type.
   * @returns The node.
   * @raises `AssertionError` if the node does not exist or if a `type` is provided but the node is not of the expected type.
   */
  getNode<T extends InvocationType>(id: string, type?: T): Invocation<T> {
    const node = this._graph.nodes[id];
    assert(node !== undefined, Graph.getNodeNotFoundMsg(id));
    if (type) {
      assert(node.type === type, Graph.getNodeNotOfTypeMsg(node, type));
    }
    // We just asserted that the node type is correct, this is OK to cast
    return node as Invocation<T>;
  }

  /**
   * Gets a node from the graph without raising an error if the node does not exist or is not of the expected type.
   * @param id The id of the node to get.
   * @param type The type of the node to get. If provided, node is guaranteed to be of this type.
   * @returns The node, if it exists and is of the correct type. Otherwise, `undefined`.
   */
  getNodeSafe<T extends InvocationType>(id: string, type?: T): Invocation<T> | undefined {
    try {
      return this.getNode(id, type);
    } catch {
      return undefined;
    }
  }

  /**
   * Update a node in the graph. Properties are shallow-copied from `updates` to the node.
   * @param id The id of the node to update.
   * @param type The type of the node to update. If provided, node is guaranteed to be of this type.
   * @param updates The fields to update on the node.
   * @returns The updated node.
   * @raises `AssertionError` if the node does not exist or its type doesn't match.
   */
  updateNode<T extends InvocationType>(id: string, type: T, updates: Partial<Invocation<T>>): Invocation<T> {
    const node = this.getNode(id, type);
    Object.assign(node, updates);
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
  getIncomers(nodeId: string): AnyInvocation[] {
    return this.getEdgesTo(nodeId).map((edge) => this.getNode(edge.source.node_id));
  }

  /**
   * Get the immediate outgoers of a node.
   * @param nodeId The id of the node to get the outgoers of.
   * @returns The outgoing nodes.
   * @raises `AssertionError` if the node does not exist.
   */
  getOutgoers(nodeId: string): AnyInvocation[] {
    return this.getEdgesFrom(nodeId).map((edge) => this.getNode(edge.destination.node_id));
  }
  //#endregion

  //#region Edge Operations

  /**
   * Add an edge to the graph. If an edge with the same source and destination already exists, an `AssertionError` is raised.
   * Provide the from and to node types as generics to get type hints for from and to field names.
   * @param fromNodeId The id of the source node.
   * @param fromField The field of the source node.
   * @param toNodeId The id of the destination node.
   * @param toField The field of the destination node.
   * @returns The added edge.
   * @raises `AssertionError` if an edge with the same source and destination already exists.
   */
  addEdge<TFrom extends EdgeNodeType, TTo extends EdgeNodeType>(
    fromNodeId: string,
    fromField: EdgeFromField<TFrom>,
    toNodeId: string,
    toField: EdgeToField<TTo>
  ): Edge {
    const edge = {
      source: { node_id: fromNodeId, field: fromField },
      destination: { node_id: toNodeId, field: toField },
    };
    assert(
      !this._graph.edges.some((e) => isEqual(e, edge)),
      Graph.getEdgeAlreadyExistsMsg(fromNodeId, fromField, toNodeId, toField)
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
  getEdge<TFrom extends EdgeNodeType, TTo extends EdgeNodeType>(
    fromNode: string,
    fromField: EdgeFromField<TFrom>,
    toNode: string,
    toField: EdgeToField<TTo>
  ): Edge {
    const edge = this._graph.edges.find(
      (e) =>
        e.source.node_id === fromNode &&
        e.source.field === fromField &&
        e.destination.node_id === toNode &&
        e.destination.field === toField
    );
    assert(edge !== undefined, Graph.getEdgeNotFoundMsg(fromNode, fromField, toNode, toField));
    return edge;
  }

  /**
   * Get an edge from the graph, or undefined if it doesn't exist.
   * Provide the from and to node types as generics to get type hints for from and to field names.
   * @param fromNodeId The id of the source node.
   * @param fromField The field of the source node.
   * @param toNodeId The id of the destination node.
   * @param toField The field of the destination node.
   * @returns The edge, or undefined if it doesn't exist.
   */
  getEdgeSafe<TFrom extends EdgeNodeType, TTo extends EdgeNodeType>(
    fromNode: string,
    fromField: EdgeFromField<TFrom>,
    toNode: string,
    toField: EdgeToField<TTo>
  ): Edge | undefined {
    try {
      return this.getEdge(fromNode, fromField, toNode, toField);
    } catch {
      return undefined;
    }
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

  hasEdge<TFrom extends EdgeNodeType, TTo extends EdgeNodeType>(
    fromNode: string,
    fromField: EdgeFromField<TFrom>,
    toNode: string,
    toField: EdgeToField<TTo>
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
  getEdgesFrom<TFrom extends EdgeNodeType>(fromNodeId: string, fromField?: EdgeFromField<TFrom>): Edge[] {
    let edges = this._graph.edges.filter((edge) => edge.source.node_id === fromNodeId);
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
  getEdgesTo<TTo extends EdgeNodeType>(toNodeId: string, toField?: EdgeToField<TTo>): Edge[] {
    let edges = this._graph.edges.filter((edge) => edge.destination.node_id === toNodeId);
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
   * @param toNodeId The id of the destination node.
   * @param toField The field of the destination node (optional).
   */
  deleteEdgesTo<TTo extends EdgeNodeType>(toNodeId: string, toField?: EdgeToField<TTo>): void {
    for (const edge of this.getEdgesTo<TTo>(toNodeId, toField)) {
      this._deleteEdge(edge);
    }
  }

  /**
   * Delete all edges from a node. If `fromField` is provided, only edges from that field are deleted.
   * Provide the from node type as a generic to get type hints for from field names.
   * @param toNodeId The id of the source node.
   * @param toField The field of the source node (optional).
   */
  deleteEdgesFrom<TFrom extends EdgeNodeType>(fromNodeId: string, fromField?: EdgeFromField<TFrom>): void {
    for (const edge of this.getEdgesFrom<TFrom>(fromNodeId, fromField)) {
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

  static uuid = uuidv4;
  //#endregion
}
