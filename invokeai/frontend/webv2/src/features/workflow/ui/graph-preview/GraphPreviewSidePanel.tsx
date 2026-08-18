import type { GraphPreviewSourceState } from '@features/workflow/ui/contracts';

import { DataList, Stack, Text } from '@chakra-ui/react';
import { Scrollable } from '@platform/ui';
import { useTranslation } from 'react-i18next';

/**
 * The preview dialog's right-hand rail: a quick summary of the compiled
 * graph (node count, result destination, source-specific settings rows).
 * Task 6 extends this with a node-selection state that replaces the summary
 * with the selected node's field values.
 */
export const GraphPreviewSidePanel = ({ source }: { source: GraphPreviewSourceState }) => {
  const { t } = useTranslation();
  const nodeCount = source.graph?.nodes.length;

  return (
    <Scrollable flexShrink={0} h="full" label={t('graphPreview.thisGraph')} w="19rem">
      <Stack gap="2" p="2">
        <Stack gap="0.5">
          <Text fontSize="sm" fontWeight="semibold">
            {t('graphPreview.thisGraph')}
          </Text>
          <Text color="fg.muted" fontSize="2xs">
            {t('graphPreview.selectNode')}
          </Text>
        </Stack>
        <DataList.Root gap="1.5" orientation="horizontal" size="sm">
          <DataList.Item>
            <DataList.ItemLabel fontSize="2xs">{t('graphPreview.nodes')}</DataList.ItemLabel>
            <DataList.ItemValue fontSize="2xs" minW="0">
              {nodeCount !== undefined ? String(nodeCount) : '—'}
            </DataList.ItemValue>
          </DataList.Item>
          <DataList.Item>
            <DataList.ItemLabel fontSize="2xs">{t('graphPreview.destination')}</DataList.ItemLabel>
            <DataList.ItemValue fontSize="2xs" minW="0">
              {source.destinationLabel ?? '—'}
            </DataList.ItemValue>
          </DataList.Item>
          {source.summaryRows.map((row) => (
            <DataList.Item key={row.id}>
              <DataList.ItemLabel fontSize="2xs">{row.label}</DataList.ItemLabel>
              <DataList.ItemValue fontSize="2xs" minW="0">
                {row.value}
              </DataList.ItemValue>
            </DataList.Item>
          ))}
        </DataList.Root>
      </Stack>
    </Scrollable>
  );
};
