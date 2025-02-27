import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Text } from '@invoke-ai/ui-library';
import type { DndTargetState } from 'features/dnd/types';
import { isNil, isString } from 'lodash-es';
import type { ReactNode } from 'react';
import { memo } from 'react';

type Props = {
  dndState: DndTargetState;
  label?: string | ReactNode;
  withBackdrop?: boolean;
};

const sx = {
  position: 'absolute',
  top: 0,
  right: 0,
  bottom: 0,
  left: 0,
  color: 'base.300',
  borderColor: 'base.300',
  transitionProperty: 'common',
  transitionDuration: '0.1s',
  '.dnd-drop-overlay-backdrop': {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    bg: 'base.900',
    opacity: 0.7,
    borderRadius: 'base',
    alignItems: 'center',
    justifyContent: 'center',
    transitionProperty: 'inherit',
    transitionDuration: 'inherit',
  },
  '.dnd-drop-overlay-content': {
    position: 'absolute',
    top: 0.5,
    right: 0.5,
    bottom: 0.5,
    left: 0.5,
    opacity: 1,
    borderWidth: 1.5,
    borderRadius: 'base',
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center',
    borderColor: 'inherit',
    transitionProperty: 'inherit',
    transitionDuration: 'inherit',
  },
  '.dnd-drop-overlay-label': {
    fontSize: 'lg',
    fontWeight: 'semibold',
    textAlign: 'center',
    color: 'inherit',
    transitionProperty: 'inherit',
    transitionDuration: 'inherit',
  },
  '&[data-dnd-state="over"]': {
    color: 'invokeYellow.300',
    borderColor: 'invokeYellow.300',
  },
} satisfies SystemStyleObject;

export const DndDropOverlay = memo((props: Props) => {
  const { dndState, label, withBackdrop = true } = props;

  if (dndState === 'idle') {
    return null;
  }

  return (
    <Flex className="dnd-drop-overlay" data-dnd-state={dndState} sx={sx}>
      {withBackdrop && <Flex className="dnd-drop-overlay-backdrop" />}
      <Flex className="dnd-drop-overlay-content">
        {isString(label) && <Text className="dnd-drop-overlay-label">{label}</Text>}
        {!isNil(label) && !isString(label) && label}
      </Flex>
    </Flex>
  );
});

DndDropOverlay.displayName = 'DndDropOverlay';
