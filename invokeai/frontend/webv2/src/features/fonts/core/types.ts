/** A variation axis exposed by a font file. */
export interface FontAxis {
  tag: string;
  label: string;
  minimum: number;
  default: number;
  maximum: number;
  hidden: boolean;
}

/** A named variation instance supplied by the font's metadata. */
export interface FontInstance {
  name: string;
  coordinates: Readonly<Record<string, number>>;
}

export type FontScope = 'private' | 'shared';
export type FontSource = 'uploaded' | 'directory';

/** id and contentHash identify a font face; family and filename are display metadata. */
export interface FontRecord {
  id: string;
  family: string;
  label: string;
  style: string;
  weight: number;
  contentHash: string;
  scope: FontScope;
  source: FontSource;
  filename: string;
  byteSize: number;
  url: string;
  axes: readonly FontAxis[];
  instances: readonly FontInstance[];
}

export interface FontCatalogPage {
  items: readonly FontRecord[];
  total: number;
  offset: number;
  limit: number;
}

/** Persistable input for resolving an indexed font in the browser runtime. */
export interface FontReference {
  id: string;
  contentHash?: string | null;
  family?: string;
  style?: string;
  weight?: number;
  axes?: Readonly<Record<string, number>>;
}

export interface FontDownloadReference extends FontReference {
  /** A non-empty axis map requests a pinned static instance from the backend. */
  axes?: Readonly<Record<string, number>>;
}

export type FontLoadState = 'idle' | 'loading' | 'ready' | 'error';
