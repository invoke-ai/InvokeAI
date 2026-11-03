/**
 * Why the document model refuses a command regardless of runtime state. Runtime transaction
 * outcomes live in `canvas-engine/editConcurrency.ts`; this vocabulary must never depend on them.
 */
export type CanvasCommandRefusal = 'missing' | 'locked' | 'invalid-target' | 'wrong-type' | 'unsupported';
