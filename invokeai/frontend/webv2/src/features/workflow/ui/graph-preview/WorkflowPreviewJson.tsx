import { Box } from '@chakra-ui/react';
import { JsonPreview } from '@platform/ui/JsonPreview';
import { useMemo } from 'react';

/**
 * The stage's JSON view: the JSON the stage's menu copies and downloads, serialized when the view shows. Loaded on
 * demand (`WorkflowPreviewJson` in `./WorkflowPreviewStage`).
 */
const WorkflowPreviewJsonView = ({
  copyFailedLabel,
  copyLabel,
  getText,
  label,
}: {
  copyFailedLabel: string;
  copyLabel: string;
  /** Serializes the JSON; called once per value it closes over. */
  getText: () => string;
  /** Names the JSON's scroll region. */
  label: string;
}) => {
  const text = useMemo(() => getText(), [getText]);

  return (
    <Box data-preview-json="" h="full">
      <JsonPreview
        copyFailedLabel={copyFailedLabel}
        copyLabel={copyLabel}
        h="full"
        label={label}
        maxH="100%"
        text={text}
      />
    </Box>
  );
};

export default WorkflowPreviewJsonView;
