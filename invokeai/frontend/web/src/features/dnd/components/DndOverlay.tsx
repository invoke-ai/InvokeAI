import { DragOverlay } from '@dnd-kit/core';
import { useScaledModifer } from 'features/dnd/hooks/useScaledCenteredModifer';
import type { TypesafeDraggableData } from 'features/dnd/types';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';

import DragPreview from './DragPreview';

type DndOverlayProps = {
  activeDragData: TypesafeDraggableData | null;
};

const DndOverlay = (props: DndOverlayProps) => {
  const scaledModifier = useScaledModifer();
  const modifiers = useMemo(() => [scaledModifier], [scaledModifier]);

  return (
    <DragOverlay dropAnimation={null} modifiers={modifiers} style={dragOverlayStyles}>
      <AnimatePresence>
        {props.activeDragData && (
          <motion.div layout key="overlay-drag-image" initial={initial} animate={animate}>
            <DragPreview dragData={props.activeDragData} />
          </motion.div>
        )}
      </AnimatePresence>
    </DragOverlay>
  );
};

export default memo(DndOverlay);

const dragOverlayStyles: CSSProperties = {
  width: 'min-content',
  height: 'min-content',
  cursor: 'grabbing',
  userSelect: 'none',
  // expand overlay to prevent cursor from going outside it and displaying
  padding: '10rem',
};

const initial: AnimationProps['initial'] = {
  opacity: 0,
  scale: 0.7,
};
const animate: AnimationProps['animate'] = {
  opacity: 1,
  scale: 1,
  transition: { duration: 0.1 },
};
