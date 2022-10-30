import { FaRecycle } from 'react-icons/fa';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAIIconButton from '../../../common/components/IAIIconButton';
import { setShouldLoopback } from '../optionsSlice';

const LoopbackButton = () => {
  const dispatch = useAppDispatch();
  const { shouldLoopback } = useAppSelector(
    (state: RootState) => state.options
  );
  return (
    <IAIIconButton
      aria-label="Loopback"
      tooltip="Loopback"
      data-selected={shouldLoopback}
      icon={<FaRecycle />}
      onClick={() => {
        dispatch(setShouldLoopback(!shouldLoopback));
      }}
    />
  );
};

export default LoopbackButton;
