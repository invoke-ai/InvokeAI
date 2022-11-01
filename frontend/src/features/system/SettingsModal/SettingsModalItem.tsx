import { useAppDispatch } from '../../../app/store';
import IAISelect from '../../../common/components/IAISelect';
import IAISwitch from '../../../common/components/IAISwitch';

export function SettingsModalItem({
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


export function SettingsModalSelectItem({
  settingTitle,
  validValues,
  defaultValue,
  dispatcher,
}: {
  settingTitle: string;
  validValues: 
      Array<number | string>
    | Array<{ key: string; value: string | number }>;
  defaultValue: string;
  dispatcher: any;
}) {
  const dispatch = useAppDispatch();
  return (
    <IAISelect
      styleClass="settings-modal-item"
      label={settingTitle}
      validValues={validValues}
      defaultValue={defaultValue}
      onChange={(e) => dispatch(dispatcher(e.target.value))}
    />
  );
}

