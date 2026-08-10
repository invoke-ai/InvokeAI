// Keep both values in sync with src/features/nodes/ui/manager/layoutConstants.ts —
// the two managers share their layout rhythm, but cross-feature imports are
// forbidden, so the values are duplicated by convention.
export const LIBRARY_WIDTH = 'clamp(22rem, 32vw, 28rem)';

// Both column headers share a min height so the library title bar lines up with
// the detail tab bar.
export const HEADER_MIN_HEIGHT = '2.75rem';
