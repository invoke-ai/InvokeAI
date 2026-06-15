import { Box, type BoxProps, type RecipeVariantProps, useRecipe } from '@chakra-ui/react';

import { panelRecipe } from '@theme/recipes';

export type PanelProps = BoxProps & RecipeVariantProps<typeof panelRecipe>;

/**
 * Bordered surface container backed by `panelRecipe` — the one styling
 * contract for panels, cards, and wells across the workbench.
 *
 * - `tone`: `surface` (default) | `raised` | `inset` | `control`
 * - `density`: `none` (default) | `sm` | `md` — padding + gap presets
 */
export const Panel = ({ css, density, tone, ...rest }: PanelProps) => {
  const recipe = useRecipe({ recipe: panelRecipe });

  return <Box css={[recipe({ density, tone }), css]} {...rest} />;
};
