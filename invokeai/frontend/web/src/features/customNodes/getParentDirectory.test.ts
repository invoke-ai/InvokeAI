import { describe, expect, it } from 'vitest';

import { getParentDirectory } from './getParentDirectory';

describe('getParentDirectory', () => {
  it('extracts the parent from a POSIX path', () => {
    expect(getParentDirectory('/home/user/nodes/my_pack')).toBe('/home/user/nodes');
  });

  it('extracts the parent from a Windows backslash path', () => {
    expect(getParentDirectory('C:\\Users\\user\\nodes\\my_pack')).toBe('C:\\Users\\user\\nodes');
  });

  it('handles mixed separators by using the last one', () => {
    expect(getParentDirectory('C:/Users/user\\nodes\\my_pack')).toBe('C:/Users/user\\nodes');
  });

  it('returns null when there is no separator', () => {
    expect(getParentDirectory('my_pack')).toBeNull();
  });

  it('returns null for the empty string', () => {
    expect(getParentDirectory('')).toBeNull();
  });
});
