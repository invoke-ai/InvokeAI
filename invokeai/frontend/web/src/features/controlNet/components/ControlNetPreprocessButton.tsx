import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { memo, useCallback } from 'react';
import { useControlAdapterControlImage } from '../hooks/useControlAdapterControlImage';
import { controlAdapterImageProcessed } from '../store/actions';

type Props = {
  id: string;
};

const ControlNetPreprocessButton = ({ id }: Props) => {
  const controlImage = useControlAdapterControlImage(id);
  const dispatch = useAppDispatch();
  const isReady = useIsReadyToEnqueue();

  const handleProcess = useCallback(() => {
    dispatch(
      controlAdapterImageProcessed({
        id,
      })
    );
  }, [id, dispatch]);

  return (
    <IAIButton
      size="sm"
      onClick={handleProcess}
      isDisabled={Boolean(!controlImage) || !isReady}
    >
      Preprocess
    </IAIButton>
  );
};

export default memo(ControlNetPreprocessButton);
