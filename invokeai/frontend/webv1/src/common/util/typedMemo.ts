/* eslint-disable @typescript-eslint/no-explicit-any */
import type React from 'react';
import { memo } from 'react';

/**
 * A typed version of React.memo, useful for components that take generics.
 */
export const typedMemo: <T extends keyof React.JSX.IntrinsicElements | React.JSXElementConstructor<any>>(
  component: T,
  propsAreEqual?: (prevProps: React.ComponentProps<T>, nextProps: React.ComponentProps<T>) => boolean
) => T & { displayName?: string } = memo;
