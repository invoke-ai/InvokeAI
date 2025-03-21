import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, FormControl, FormHelperText } from '@invoke-ai/ui-library';
import { linkifyOptions, linkifySx } from 'common/components/linkify';
import { InputFieldRenderer } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { NodeFieldElementLabel } from 'features/nodes/components/sidePanel/builder/NodeFieldElementLabel';
import { useInputFieldDescriptionSafe } from 'features/nodes/hooks/useInputFieldDescriptionSafe';
import { useInputFieldTemplateOrThrow, useInputFieldTemplateSafe } from 'features/nodes/hooks/useInputFieldTemplate';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { NODE_FIELD_CLASS_NAME } from 'features/nodes/types/workflow';
import Linkify from 'linkify-react';
import { memo, useMemo } from 'react';

const sx: SystemStyleObject = {
  '&[data-parent-layout="column"]': {
    w: 'full',
    h: 'min-content',
  },
  '&[data-parent-layout="row"]': {
    flex: '1 1 0',
    minW: 32,
  },
  '&[data-with-description="false"]': {
    pb: 2,
  },
};

export const NodeFieldElementViewMode = memo(({ el }: { el: NodeFieldElement }) => {
  const { id, data } = el;
  const { fieldIdentifier, showDescription } = data;
  const description = useInputFieldDescriptionSafe(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const fieldTemplate = useInputFieldTemplateSafe(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const containerCtx = useContainerContext();

  const _description = useMemo(
    () => description || fieldTemplate?.description || '',
    [description, fieldTemplate?.description]
  );

  return (
    <Flex
      id={id}
      className={NODE_FIELD_CLASS_NAME}
      sx={sx}
      data-parent-layout={containerCtx.layout}
      data-with-description={showDescription && !!_description}
    >
      <NodeFieldElementViewModeContent el={el} />
    </Flex>
  );
});
NodeFieldElementViewMode.displayName = 'NodeFieldElementViewMode';

const NodeFieldElementViewModeContent = memo(({ el }: { el: NodeFieldElement }) => {
  const { data } = el;
  const { fieldIdentifier, showDescription } = data;
  const description = useInputFieldDescriptionSafe(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const fieldTemplate = useInputFieldTemplateOrThrow(fieldIdentifier.nodeId, fieldIdentifier.fieldName);

  const _description = useMemo(
    () => description || fieldTemplate.description,
    [description, fieldTemplate.description]
  );

  return (
    <FormControl flex="1 1 0" orientation="vertical">
      <NodeFieldElementLabel el={el} />
      <Flex w="full" gap={4}>
        <InputFieldRenderer
          nodeId={fieldIdentifier.nodeId}
          fieldName={fieldIdentifier.fieldName}
          settings={data.settings}
        />
      </Flex>
      {showDescription && _description && (
        <FormHelperText sx={linkifySx}>
          <Linkify options={linkifyOptions}>{_description}</Linkify>
        </FormHelperText>
      )}
    </FormControl>
  );
});
NodeFieldElementViewModeContent.displayName = 'NodeFieldElementViewModeContent';
