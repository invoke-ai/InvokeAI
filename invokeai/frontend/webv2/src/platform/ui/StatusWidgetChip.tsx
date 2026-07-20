import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

import { HStack, Icon, Text, type RecipeVariantProps, useRecipe } from '@chakra-ui/react';
import { chipRecipe } from '@theme/recipes';

export const StatusWidgetChip = ({
  children,
  icon,
  tone,
}: {
  children: ReactNode;
  icon: LucideIcon;
  tone?: NonNullable<RecipeVariantProps<typeof chipRecipe>>['tone'];
}) => {
  const recipe = useRecipe({ recipe: chipRecipe });

  return (
    <HStack css={recipe({ tone })}>
      <Icon as={icon} boxSize="3" />
      <Text whiteSpace="nowrap">{children}</Text>
    </HStack>
  );
};
