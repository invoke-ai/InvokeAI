import { memo } from 'react';

/**
 * A typed version of React.memo, useful for components that take generics.
 */
export const typedMemo: <T>(c: T) => T = memo;
