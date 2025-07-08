declare module 'elkjs/lib/elk.bundled.js' {
  export class ElkLayout {
    layout(graph: ElkNode): Promise<ElkNode>;
  }
  export interface ElkNode {
    id: string;
    width: number;
    height: number;
    x?: number;
    y?: number;
    children?: ElkNode[];
    edges?: ElkEdge[];
    layoutOptions?: Record<string, string>;
  }
  export interface ElkEdge {
    id: string;
    sources: string[];
    targets: string[];
  }
}
