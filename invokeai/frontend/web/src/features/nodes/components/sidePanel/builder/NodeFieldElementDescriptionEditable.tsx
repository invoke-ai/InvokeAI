import { FormHelperText, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { linkifyOptions, linkifySx } from 'common/components/linkify';
import { useEditable } from 'common/hooks/useEditable';
import { useInputFieldTemplateOrThrow } from 'features/nodes/hooks/useInputFieldTemplateOrThrow';
import { useInputFieldUserDescriptionSafe } from 'features/nodes/hooks/useInputFieldUserDescriptionSafe';
import { fieldDescriptionChanged } from 'features/nodes/store/nodesSlice';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import Linkify from 'linkify-react';
import { memo, useCallback, useRef } from 'react';

export const NodeFieldElementDescriptionEditable = memo(({ el }: { el: NodeFieldElement }) => {
  const { data } = el;
  const { fieldIdentifier } = data;
  const dispatch = useAppDispatch();
  const description = useInputFieldUserDescriptionSafe(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const fieldTemplate = useInputFieldTemplateOrThrow(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const onChange = useCallback(
    (description: string) => {
      dispatch(
        fieldDescriptionChanged({
          nodeId: fieldIdentifier.nodeId,
          fieldName: fieldIdentifier.fieldName,
          val: description,
        })
      );
    },
    [dispatch, fieldIdentifier.fieldName, fieldIdentifier.nodeId]
  );

  const editable = useEditable({
    value: description || fieldTemplate.description,
    defaultValue: fieldTemplate.description,
    inputRef,
    onChange,
  });

  if (!editable.isEditing) {
    return (
      <FormHelperText onDoubleClick={editable.startEditing} sx={linkifySx}>
        <Linkify options={linkifyOptions}>{editable.value}</Linkify>
      </FormHelperText>
    );
  }

  return (
    <Textarea
      ref={inputRef}
      variant="outline"
      fontSize="sm"
      p={1}
      px={2}
      _focusVisible={{ borderRadius: 'base', h: 'unset' }}
      {...editable.inputProps}
    />
  );
});
NodeFieldElementDescriptionEditable.displayName = 'NodeFieldElementDescriptionEditable';
