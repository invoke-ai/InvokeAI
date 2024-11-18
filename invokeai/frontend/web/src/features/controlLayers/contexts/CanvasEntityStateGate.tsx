import { useAppSelector } from 'app/store/storeHooks';
import { selectEntityExists } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';

/**
 * A "gate" component that renders its children only if the entity exists in redux state.
 */
export const CanvasEntityStateGate = memo((props: PropsWithChildren<{ entityIdentifier: CanvasEntityIdentifier }>) => {
  const selector = useMemo(() => selectEntityExists(props.entityIdentifier), [props.entityIdentifier]);
  const entityExistsInState = useAppSelector(selector);

  if (!entityExistsInState) {
    return null;
  }

  return props.children;
});
CanvasEntityStateGate.displayName = 'CanvasEntityStateGate';
