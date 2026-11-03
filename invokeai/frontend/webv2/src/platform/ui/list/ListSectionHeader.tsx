import { chakra, useSlotRecipe } from '@chakra-ui/react';
import { listSectionHeaderSlotRecipe } from '@theme/recipes';
import { useMemo } from 'react';

/** Fixed so virtualized lists can place headers without measuring them. */
export const LIST_SECTION_HEADER_HEIGHT_PX = 32;

export const ListSectionHeader = ({ count, label }: { count?: number; label: string }) => {
  const recipe = useSlotRecipe({ recipe: listSectionHeaderSlotRecipe });
  const styles = useMemo(() => recipe({}), [recipe]);

  return (
    <chakra.div css={styles.root}>
      <chakra.span css={styles.label}>{label}</chakra.span>
      {count !== undefined ? <chakra.span css={styles.count}>{count}</chakra.span> : null}
    </chakra.div>
  );
};
