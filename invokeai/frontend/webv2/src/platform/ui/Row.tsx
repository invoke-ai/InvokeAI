import { Box, type BoxProps, type RecipeVariantProps, useRecipe } from '@chakra-ui/react';
import { rowRecipe } from '@theme/recipes';
import { useMemo } from 'react';

export type RowProps = BoxProps & RecipeVariantProps<typeof rowRecipe>;

/** Use as=button for clickable rows; active selects the themed emphasis level. */
export const Row = ({ active, css, ...rest }: RowProps) => {
  const recipe = useRecipe({ recipe: rowRecipe });
  const rowCss = useMemo(() => [recipe({ active }), css], [active, css, recipe]);

  return <Box css={rowCss} {...rest} />;
};
