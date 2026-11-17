import { chakra, useSlotRecipe } from '@chakra-ui/react';
import { listSectionHeaderSlotRecipe } from '@theme/recipes';
import { useMemo } from 'react';

/** Fixed so virtualized lists can place headers without measuring them. */
export const LIST_SECTION_HEADER_HEIGHT_PX = 32;

export const ListSectionHeader = ({
  count,
  label,
  placement = 'list',
}: {
  count?: number;
  label: string;
  /** `bar` centres the label for a header row that also holds controls; `list` sits it low, above its rows. */
  placement?: 'bar' | 'list';
}) => {
  const recipe = useSlotRecipe({ recipe: listSectionHeaderSlotRecipe });
  const styles = useMemo(() => recipe({ placement }), [placement, recipe]);

  return (
    <chakra.div css={styles.root}>
      <chakra.span css={styles.label}>{label}</chakra.span>
      {count !== undefined ? <chakra.span css={styles.count}>{count}</chakra.span> : null}
    </chakra.div>
  );
};
