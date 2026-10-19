import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

import { Box, HStack, Icon, Text, type RecipeVariantProps, type SystemStyleObject, useRecipe } from '@chakra-ui/react';
import { chipRecipe } from '@theme/recipes';

/** Presence turns on the bar; a null `value` renders it as a full quiet fill. */
export interface StatusWidgetChipProgress {
  value: number | null;
}

const PROGRESS_TRACK_SX: SystemStyleObject = {
  bottom: 0,
  height: '2px',
  insetInline: 0,
  pointerEvents: 'none',
  position: 'absolute',
};

const PROGRESS_FILL_SX: SystemStyleObject = {
  bg: 'accent.solid',
  height: 'full',
  transition: 'width var(--wb-motion-duration-fast) linear',
};

/** Use a static fill under reduced motion; otherwise sweep the indeterminate band. */
const PROGRESS_SWEEP_SX: SystemStyleObject = {
  ':root[data-reduce-motion="true"] &': { animationName: 'none' },
  animationDuration: '1.4s',
  animationIterationCount: 'infinite',
  animationName: 'wb-status-sweep',
  animationTimingFunction: 'ease-in-out',
  bg: 'accent.solid',
  height: 'full',
  width: '33%',
};

export const StatusWidgetChip = ({
  children,
  icon,
  progress,
  tone,
}: {
  children: ReactNode;
  icon: LucideIcon;
  /** Optional live progress, drawn as a hairline along the chip's bottom edge. */
  progress?: StatusWidgetChipProgress;
  tone?: NonNullable<RecipeVariantProps<typeof chipRecipe>>['tone'];
}) => {
  const recipe = useRecipe({ recipe: chipRecipe });

  return (
    <HStack css={recipe({ tone })} position={progress ? 'relative' : undefined}>
      <Icon as={icon} boxSize="3" />
      <Text whiteSpace="nowrap">{children}</Text>
      {progress ? (
        <Box aria-hidden="true" css={PROGRESS_TRACK_SX} overflow="hidden">
          {progress.value === null ? (
            <>
              <Box css={PROGRESS_FILL_SX} opacity={0.35} position="absolute" width="full" />
              <Box css={PROGRESS_SWEEP_SX} position="absolute" />
            </>
          ) : (
            <Box css={PROGRESS_FILL_SX} width={`${progress.value * 100}%`} />
          )}
        </Box>
      ) : null}
    </HStack>
  );
};
