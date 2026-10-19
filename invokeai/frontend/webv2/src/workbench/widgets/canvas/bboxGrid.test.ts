import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { DEFAULT_MODEL_GRID, gridSizeForModelBase, resolveModelGrid } from './bboxGrid';

describe('gridSizeForModelBase', () => {
  seedArchitectureCapabilities();

  it('reads the grid the architecture declares', () => {
    // The same number the denoise node enforces through `multiple_of` on width/height.
    for (const base of ['flux', 'flux2', 'sd-3', 'qwen-image', 'z-image', 'ernie-image']) {
      expect(gridSizeForModelBase(base)).toBe(16);
    }

    expect(gridSizeForModelBase('cogview4')).toBe(32);

    for (const base of ['sd-1', 'sd-2', 'sdxl', 'anima']) {
      expect(gridSizeForModelBase(base)).toBe(8);
    }
  });

  it('reads the 32px grid Wan TI2V-5B declares, from the table the backend really serves', () => {
    // Use the committed variant fixture: 1280x720 passes base grid 16 but fails TI2V-5B grid 32.
    expect(gridSizeForModelBase('wan', 'ti2v_5b')).toBe(32);
    expect(gridSizeForModelBase('wan')).toBe(16);
    expect(720 % gridSizeForModelBase('wan', 'ti2v_5b')).not.toBe(0);
  });

  it('no longer offers 8px steps for architectures that reject them', () => {
    for (const base of ['krea-2', 'wan', 'ideogram-4']) {
      expect(gridSizeForModelBase(base)).toBe(16);
    }
  });

  it('falls back to the default grid when there is no model or no row for its architecture', () => {
    expect(gridSizeForModelBase(null)).toBe(DEFAULT_MODEL_GRID);
    expect(gridSizeForModelBase(undefined)).toBe(DEFAULT_MODEL_GRID);
    expect(gridSizeForModelBase('mystery-model')).toBe(DEFAULT_MODEL_GRID);
    expect(DEFAULT_MODEL_GRID).toBe(8);
  });
});

describe('resolveModelGrid before the capability table arrives', () => {
  // Leave capabilities absent to reproduce reopening before the table arrives.

  it('reports the grid as unknown rather than as 8', () => {
    // The distinction a writer needs: 8 is a real answer for SDXL and a guess for Wan, and only
    // one of the two may be persisted into a project file.
    expect(resolveModelGrid('wan')).toBeNull();
    expect(resolveModelGrid('cogview4')).toBeNull();
    expect(resolveModelGrid('sdxl')).toBeNull();
  });

  it('still answers for "no model selected", which the backend has no say in', () => {
    expect(resolveModelGrid(null)).toBe(DEFAULT_MODEL_GRID);
    expect(resolveModelGrid(undefined)).toBe(DEFAULT_MODEL_GRID);
  });

  it('answers for an external generator too, which will never get a row to wait for', () => {
    // A writer holds off on `null`, so answering it here would stop the bbox <-> dims sync for
    // external providers for the whole session.
    expect(resolveModelGrid('external')).toBe(DEFAULT_MODEL_GRID);
  });

  it('keeps the default for readers that must show something', () => {
    expect(gridSizeForModelBase('wan')).toBe(DEFAULT_MODEL_GRID);
  });
});

describe('the grid a variant declares', () => {
  // Override variant policy independently of fixtures: Wan grid depends on base plus variant, and 720 fails grid
  // 32 despite passing 16.
  beforeEach(() => {
    setArchitectureCapabilities(
      architectureCapabilitiesFixture.map((row) =>
        row.base === 'wan' && row.variant === 'ti2v_5b'
          ? { ...row, features: { ...row.features, dimension_grid: 32 } }
          : row
      )
    );
  });

  afterEach(() => {
    resetArchitectureCapabilities();
  });

  it('prefers the variant row over the architecture row', () => {
    expect(gridSizeForModelBase('wan', 'ti2v_5b')).toBe(32);
    expect(gridSizeForModelBase('wan')).toBe(16);
  });

  it('falls back to the architecture row for a variant that declares nothing of its own', () => {
    expect(gridSizeForModelBase('wan', 'a14b')).toBe(16);
    expect(gridSizeForModelBase('sdxl', 'inpaint')).toBe(8);
  });
});
