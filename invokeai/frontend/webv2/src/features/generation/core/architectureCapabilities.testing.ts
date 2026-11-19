import { afterEach, beforeEach } from 'vitest';

import fixture from './__fixtures__/architectureCapabilities.json';
import {
  type ArchitectureCapabilitiesRow,
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from './architectureCapabilities';

/** The response body the backend actually serves, pinned by `test_capabilities_fixture.py`. */
export const architectureCapabilitiesFixture = fixture as ArchitectureCapabilitiesRow[];

export const seedArchitectureCapabilities = (): void => {
  beforeEach(() => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
  });

  afterEach(() => {
    resetArchitectureCapabilities();
  });
};
