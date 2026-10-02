import { Box, HStack, Text } from '@chakra-ui/react';
import { getImageCluster, parseGallerySemanticReference } from '@features/gallery/contracts';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { getClusterColor } from '@workbench/image-map/clusterPalette';
import { formatClusterSize, isClusterSizeLabel } from '@workbench/image-map/clusterSelection';
import { imageMapStore } from '@workbench/image-map/imageMapStore';
import { getDominantCluster } from '@workbench/image-map/imageMapTraces';
import { useWidgetValuesSelector } from '@workbench/WorkbenchContext';
import { XIcon } from 'lucide-react';
import { useCallback, useMemo, type MouseEvent } from 'react';

import { useClearClusterSelection } from './useSelectMapImage';

export const CLEAR_CLUSTER_SELECTION_LABEL = 'Clear cluster selection';

/** The registry key of the gallery's cluster listing, or null when it shows anything else. */
export const selectClusterSelectionId = (values: Record<string, unknown>): string | null => {
  const reference = parseGallerySemanticReference(values.semanticImageQuery);

  return reference?.kind === 'cluster' ? reference.clusterId : null;
};

/**
 * Names the selected cluster over the dimmed map and offers the way back to full colour. It is what explains the grey:
 * without it a dimmed map reads as a rendering fault, and the only clear lived in the Gallery's search field.
 */
export const ClusterSelectionChip = () => {
  const clusterId = useWidgetValuesSelector('gallery', selectClusterSelectionId);
  const clearClusterSelection = useClearClusterSelection();
  const handleClear = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      // The button unmounts with the chip; hand focus to the map instead of
      // letting it fall to the page.
      event.currentTarget.closest<HTMLElement>('[data-image-map-surface]')?.focus();
      clearClusterSelection();
    },
    [clearClusterSelection]
  );
  const points = imageMapStore.useSelector((snapshot) => snapshot.data?.points ?? null);
  // The registry entry behind an id never changes, so the id is its identity.
  const cluster = clusterId ? getImageCluster(clusterId) : null;
  const color = useMemo(() => {
    if (!cluster || !points) {
      return null;
    }

    const dominant = getDominantCluster(points, new Set(cluster.itemKeys));

    return dominant === null ? null : getClusterColor(dominant);
  }, [cluster, points]);

  if (!cluster) {
    return null;
  }

  const size = formatClusterSize(cluster.itemKeys.length);
  const label = isClusterSizeLabel(cluster.label) ? null : cluster.label;

  return (
    <HStack
      aria-label="Selected cluster"
      bg="bg.panel"
      borderColor="border.subtle"
      borderRadius="full"
      borderWidth="1px"
      boxShadow="sm"
      fontSize="xs"
      gap="1.5"
      maxW="full"
      minW="0"
      pe="0.5"
      pointerEvents="auto"
      ps="2.5"
      py="0.5"
      role="group"
    >
      {/* No swatch when none of the members is on the map any more: there is no colour to match. */}
      {color ? <Box aria-hidden="true" bg={color} borderRadius="full" boxSize="2.5" flexShrink={0} /> : null}
      {label ? (
        <Tooltip content={label}>
          <Text color="fg" minW="0" truncate>
            {label}
          </Text>
        </Tooltip>
      ) : null}
      <Text color={label ? 'fg.muted' : 'fg'} flexShrink={0} fontVariantNumeric="tabular-nums">
        {label ? `· ${size}` : size}
      </Text>
      <Tooltip content={`${CLEAR_CLUSTER_SELECTION_LABEL} (Esc)`}>
        <IconButton
          aria-label={CLEAR_CLUSTER_SELECTION_LABEL}
          borderRadius="full"
          flexShrink={0}
          size="2xs"
          variant="ghost"
          onClick={handleClear}
        >
          <XIcon />
        </IconButton>
      </Tooltip>
    </HStack>
  );
};
