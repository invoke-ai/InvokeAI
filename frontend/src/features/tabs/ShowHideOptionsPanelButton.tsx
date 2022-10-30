import { IoMdOptions } from 'react-icons/io';
import { useAppDispatch } from '../../app/store';
import IAIIconButton from '../../common/components/IAIIconButton';
import { setShouldShowOptionsPanel } from '../options/optionsSlice';

const ShowHideOptionsPanelButton = () => {
  const dispatch = useAppDispatch();

  const handleShowOptionsPanel = () => {
    dispatch(setShouldShowOptionsPanel(true));
  };

  return (
    <IAIIconButton
      tooltip="Show Options Panel (G)"
      tooltipPlacement="top"
      aria-label="Show Options Panel"
      styleClass="floating-show-hide-button left"
      onMouseOver={handleShowOptionsPanel}
    >
      <IoMdOptions />
    </IAIIconButton>
  );
};

export default ShowHideOptionsPanelButton;
