import { apiFetchJson } from '@platform/transport/http';

import type { ServerOutputReadPort } from './reader';
import type { ServerOutputRead } from './transcript';

/** The cursor is opaque and bounded by the server; the browser cache never keeps a transcript. */
export const readServerOutput: ServerOutputReadPort = (cursor, signal) =>
  apiFetchJson<ServerOutputRead>(
    cursor === null ? '/api/v1/app/server_output' : `/api/v1/app/server_output?cursor=${encodeURIComponent(cursor)}`,
    { cache: 'no-store', signal }
  );
