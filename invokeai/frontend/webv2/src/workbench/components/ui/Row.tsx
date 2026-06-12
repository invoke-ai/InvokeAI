import { Box, type BoxProps, type RecipeVariantProps, useRecipe } from '@chakra-ui/react';

import { rowRecipe } from '../../../theme/recipes';

export type RowProps = BoxProps & RecipeVariantProps<typeof rowRecipe>;

/**
 * Interactive list / table row backed by `rowRecipe`: hover and focus fills
 * come from the recipe, and the previously ad-hoc `isSelected ? … : …`
 * ternaries collapse into the `active` variant.
 *
 * - `active`: `none` (default) | `muted` (selected) | `brand` (identity) |
 *   `accent` (active item / selection highlight)
 *
 * Render as a real `<button>` (`as="button"`) when the row is clickable.
 */
export const Row = ({ active, css, ...rest }: RowProps) => {
  const recipe = useRecipe({ recipe: rowRecipe });

  return <Box css={[recipe({ active }), css]} {...rest} />;
};
