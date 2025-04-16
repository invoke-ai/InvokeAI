import { forwardRef } from 'react';

/**
 * A forwardRef that works with generics and doesn't require the use of `as` to cast the type.
 * See: https://www.totaltypescript.com/forwardref-with-generic-components
 */
export function fixedForwardRef<T, P = object>(
  render: (props: P, ref: React.Ref<T>) => React.ReactNode
): (props: P & React.RefAttributes<T>) => React.ReactNode {
  // @ts-expect-error: This is a workaround for forwardRef's crappy typing
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return forwardRef(render) as any;
}
