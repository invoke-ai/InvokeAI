import { Input } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityTitle } from 'features/controlLayers/hooks/useEntityTitle';
import { entityNameChanged } from 'features/controlLayers/store/canvasV2Slice';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';

type Props = {
  onStopEditing: () => void;
};

export const CanvasEntityTitleEdit = memo(({ onStopEditing }: Props) => {
  const dispatch = useAppDispatch();
  const ref = useRef<HTMLInputElement>(null);
  const entityIdentifier = useEntityIdentifierContext();
  const title = useEntityTitle(entityIdentifier);
  const [localTitle, setLocalTitle] = useState(title);

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
    onStopEditing();
  }, [dispatch, entityIdentifier, localTitle, onStopEditing, title]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        onBlur();
      } else if (e.key === 'Escape') {
        setLocalTitle(title);
        onStopEditing();
      }
    },
    [onBlur, onStopEditing, title]
  );

  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);

  return (
    <Input ref={ref} value={localTitle} onChange={onChange} onBlur={onBlur} onKeyDown={onKeyDown} variant="outline" />
  );
});

CanvasEntityTitleEdit.displayName = 'CanvasEntityTitleEdit';
