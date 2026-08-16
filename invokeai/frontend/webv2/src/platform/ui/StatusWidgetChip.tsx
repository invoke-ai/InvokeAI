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
        <Box aria-hidden="true" css={PROGRESS_TRACK_SX}>
          <Box
            css={PROGRESS_FILL_SX}
            opacity={progress.value === null ? 0.35 : 1}
            width={`${(progress.value ?? 1) * 100}%`}
          />
        </Box>
      ) : null}
    </HStack>
  );
};
