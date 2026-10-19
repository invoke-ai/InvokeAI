import type { RowProps } from '@platform/ui';
import type { CSSProperties, ReactNode } from 'react';

import { Row } from '@platform/ui';

const LAYER_ROW_BACKGROUND_TRANSITION = 'background min(40ms, var(--wb-motion-duration-fast)) ease-out';

export const LayerRowSurface = ({
  active,
  children,
  cursor = 'default',
  indentStyle,
}: {
  active: RowProps['active'];
  children: ReactNode;
  cursor?: 'default' | 'grabbing';
  indentStyle?: CSSProperties;
}) => (
  <Row
    active={active}
    alignItems="center"
    cursor={cursor}
    display="flex"
    gap="1.5"
    h="full"
    px="1.5"
    style={indentStyle}
    transition={LAYER_ROW_BACKGROUND_TRANSITION}
  >
    {children}
  </Row>
);
