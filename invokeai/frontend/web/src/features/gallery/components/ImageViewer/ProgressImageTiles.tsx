import { Flex, Grid, GridItem } from '@invoke-ai/ui-library';
import { memo, useMemo } from 'react';

import type { ViewerProgressDatum } from './context';
import { ProgressImage } from './ProgressImage2';
import { ProgressIndicator } from './ProgressIndicator2';

/**
 * Renders one tile per concurrently-running session (multi-GPU). Each tile shows that session's live
 * preview image plus a small progress indicator. Used by the viewer when more than one session is
 * active; a single active session uses the full-size preview instead.
 */
export const ProgressImageTiles = memo(({ data }: { data: ViewerProgressDatum[] }) => {
  // Lay the tiles out in a roughly-square grid that grows with the number of active sessions.
  const columns = useMemo(() => Math.ceil(Math.sqrt(data.length)), [data.length]);

  return (
    <Grid
      w="full"
      h="full"
      gap={2}
      p={2}
      gridTemplateColumns={`repeat(${columns}, minmax(0, 1fr))`}
      gridAutoRows="1fr"
      alignItems="center"
      justifyItems="center"
    >
      {data.map((datum) => (
        <GridItem key={datum.itemId} w="full" h="full" minW={0} minH={0}>
          <Flex w="full" h="full" position="relative" alignItems="center" justifyContent="center">
            <ProgressImage progressImage={datum.progressImage} />
            <ProgressIndicator progressEvent={datum.progressEvent} position="absolute" top={2} right={2} size={6} />
          </Flex>
        </GridItem>
      ))}
    </Grid>
  );
});
ProgressImageTiles.displayName = 'ProgressImageTiles';
