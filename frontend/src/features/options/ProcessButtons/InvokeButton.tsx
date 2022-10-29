import { generateImage } from '../../../app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAIButton from '../../../common/components/IAIButton';
import useCheckParameters from '../../../common/hooks/useCheckParameters';
import { tabMap } from '../../tabs/InvokeTabs';

export default function InvokeButton() {
  const dispatch = useAppDispatch();
  const isReady = useCheckParameters();

  const activeTab = useAppSelector(
    (state: RootState) => state.options.activeTab
  );

  const handleClickGenerate = () => {
    dispatch(generateImage(tabMap[activeTab]));
  };

  return (
    <IAIButton
      label="Invoke"
      aria-label="Invoke"
      type="submit"
      isDisabled={!isReady}
      onClick={handleClickGenerate}
      className="invoke-btn"
    />
  );
}
