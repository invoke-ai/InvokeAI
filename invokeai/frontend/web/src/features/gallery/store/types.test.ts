import type { S } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, test } from 'vitest';

import type { BoardRecordOrderBy } from './types';

describe('Gallery Types', () => {
  // Ensure zod types match OpenAPI types
  test('BoardRecordOrderBy', () => {
    assert<Equals<BoardRecordOrderBy, S['BoardRecordOrderBy']>>();
  });
});
