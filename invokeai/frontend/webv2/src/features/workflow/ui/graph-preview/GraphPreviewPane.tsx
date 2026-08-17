import type { ProjectGraphState } from '@features/workflow/contracts';

import { Box } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { documentToPreviewGraph, GraphPreviewFlow } from './GraphPreviewFlow';

/**
 * The library dialog's node preview, conversion included.
 *
 * Both halves live here so the dialog can hold the whole thing behind one
 * `lazy()` boundary: `documentToPreviewGraph` shares a module with the flow
 * renderer, so importing it for the conversion alone pulled xyflow and d3
 * (~174 KB) into every editor boot through the workflow widget's host.
 */
export const GraphPreviewPane = ({ document }: { document: ProjectGraphState }) => {
  const { t } = useTranslation();
  const { graph, positionHints } = documentToPreviewGraph(document, t('widgets.labels.workflow'));

  return (
    <Box h="24rem">
      <GraphPreviewFlow graph={graph} positionHints={positionHints} />
    </Box>
  );
};
