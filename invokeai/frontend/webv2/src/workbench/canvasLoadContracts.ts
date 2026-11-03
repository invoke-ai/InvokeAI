export type CanvasVersionScope = 'state' | 'document' | 'snapshot';

export interface CanvasLoadDiagnostic {
  readonly path: string;
  readonly message: string;
}

export type CanvasLoadRefusal = (
  | { status: 'unsupported-version'; scope: CanvasVersionScope; version: number }
  | { status: 'invalid'; scope: CanvasVersionScope; diagnostics: readonly CanvasLoadDiagnostic[] }
) & { raw: unknown };

export type CanvasLoadResult<T> =
  | { status: 'loaded'; value: T; diagnostics: readonly CanvasLoadDiagnostic[] }
  | CanvasLoadRefusal;
