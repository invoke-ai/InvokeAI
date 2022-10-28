import { useAppDispatch } from '../../../app/store';
import IAISwitch from '../../../common/components/IAISwitch';

export default function SettingsModalItem({
  settingTitle,
  isChecked,
  dispatcher,
}: {
  settingTitle: string;
  isChecked: boolean;
  dispatcher: any;
}) {
  const dispatch = useAppDispatch();
  return (
    <IAISwitch
      styleClass="settings-modal-item"
      label={settingTitle}
      isChecked={isChecked}
      onChange={(e) => dispatch(dispatcher(e.target.checked))}
    />
  );
}
