import { Input } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useBoolean } from 'common/hooks/useBoolean';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityTitle } from 'features/controlLayers/hooks/useEntityTitle';
import { entityNameChanged } from 'features/controlLayers/store/canvasSlice';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';

export const CanvasEntityEditableTitle = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const title = useEntityTitle(entityIdentifier);
  const isEditing = useBoolean(false);
  const [localTitle, setLocalTitle] = useState(title);
  const ref = useRef<HTMLInputElement>(null);

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setLocalTitle(e.target.value);
  }, []);

  const onBlur = useCallback(() => {
    const trimmedTitle = localTitle.trim();
    if (trimmedTitle.length === 0) {
      dispatch(entityNameChanged({ entityIdentifier, name: null }));
    } else if (trimmedTitle !== title) {
      dispatch(entityNameChanged({ entityIdentifier, name: trimmedTitle }));
    }
    isEditing.setFalse();
  }, [dispatch, entityIdentifier, isEditing, localTitle, title]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        onBlur();
      } else if (e.key === 'Escape') {
        setLocalTitle(title);
        isEditing.setFalse();
      }
    },
    [isEditing, onBlur, title]
  );

  useEffect(() => {
    if (isEditing.isTrue) {
      ref.current?.focus();
      ref.current?.select();
    }
  }, [isEditing.isTrue]);

  if (!isEditing.isTrue) {
    return <CanvasEntityTitle cursor="text" onDoubleClick={isEditing.setTrue} />;
  }

  return (
    <Input
      ref={ref}
      value={localTitle}
      onChange={onChange}
      onBlur={onBlur}
      onKeyDown={onKeyDown}
      variant="outline"
      _focusVisible={{ borderWidth: 1, borderColor: 'invokeBlueAlpha.400', borderRadius: 'base' }}
    />
  );
});

CanvasEntityEditableTitle.displayName = 'CanvasEntityTitleEdit';
