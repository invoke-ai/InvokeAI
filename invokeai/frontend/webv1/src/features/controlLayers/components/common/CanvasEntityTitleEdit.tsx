import { Input } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityName, useEntityTypeName } from 'features/controlLayers/hooks/useEntityTitle';
import { entityNameChanged } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback, useRef } from 'react';

export const CanvasEntityEditableTitle = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const inputRef = useRef<HTMLInputElement>(null);
  const name = useEntityName(entityIdentifier);
  const typeName = useEntityTypeName(entityIdentifier.type);

  const onChange = useCallback(
    (name: string) => {
      dispatch(entityNameChanged({ entityIdentifier, name }));
    },
    [dispatch, entityIdentifier]
  );

  const editable = useEditable({
    value: name || typeName,
    defaultValue: typeName,
    onChange,
    inputRef,
  });

  if (!editable.isEditing) {
    return <CanvasEntityTitle cursor="text" onDoubleClick={editable.startEditing} />;
  }

  return (
    <Input
      ref={inputRef}
      {...editable.inputProps}
      variant="outline"
      _focusVisible={{ borderRadius: 'base', h: 'unset' }}
    />
  );
});

CanvasEntityEditableTitle.displayName = 'CanvasEntityTitleEdit';
