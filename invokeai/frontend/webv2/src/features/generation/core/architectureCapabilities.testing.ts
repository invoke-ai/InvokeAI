/** Seed fixtures explicitly so tests can still exercise unloaded policy. */

import { afterEach, beforeEach } from 'vitest';

import fixture from './__fixtures__/architectureCapabilities.json';
import {
  type ArchitectureCapabilitiesRow,
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from './architectureCapabilities';

/** The response body the backend actually serves, pinned by `test_capabilities_fixture.py`. */
export const architectureCapabilitiesFixture = fixture as ArchitectureCapabilitiesRow[];

/** Load the table before each test and drop it afterwards, so nothing leaks between files. */
export const seedArchitectureCapabilities = (): void => {
  beforeEach(() => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
  });

  afterEach(() => {
    resetArchitectureCapabilities();
  });
};
