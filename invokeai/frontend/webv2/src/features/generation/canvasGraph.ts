/**
 * Canvas graph compilation's lazy-load interface.
 *
 * Keep this separate from the shared `graph` interface: the Canvas invocation
 * runtime is loaded on demand, while generate graph helpers are needed during
 * workbench startup. Mixing both compilers in one interface pulls Canvas-only
 * processing code into the initial editor graph.
 */
export { compileCanvasGraph } from './core/canvas/compileCanvasGraph';
export type { CanvasCompileMode } from './core/canvas/types';
