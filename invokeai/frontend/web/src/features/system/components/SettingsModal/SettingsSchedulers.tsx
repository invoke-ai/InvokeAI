import {
  Menu,
  MenuButton,
  MenuItemOption,
  MenuList,
  MenuOptionGroup,
} from '@chakra-ui/react';
import { SCHEDULERS } from 'app/constants';

import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { setSchedulers } from 'features/ui/store/uiSlice';
import { isArray } from 'lodash-es';

import { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

export default function SettingsSchedulers() {
  const schedulers = useAppSelector((state: RootState) => state.ui.schedulers);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const schedulerSettingsHandler = (v: string | string[]) => {
    if (isArray(v)) dispatch(setSchedulers(v.sort()));
  };

  const renderSchedulerMenuItems = () => {
    const schedulerMenuItemsToRender: ReactNode[] = [];

    SCHEDULERS.forEach((scheduler) => {
      schedulerMenuItemsToRender.push(
        <MenuItemOption key={scheduler} value={scheduler}>
          {scheduler}
        </MenuItemOption>
      );
    });

    return schedulerMenuItemsToRender;
  };

  return (
    <Menu closeOnSelect={false}>
      <MenuButton as={IAIButton}>
        {t('settings.availableSchedulers')}
      </MenuButton>
      <MenuList minWidth="480px" maxHeight="39vh" overflowY="scroll">
        <MenuOptionGroup
          value={schedulers}
          type="checkbox"
          onChange={schedulerSettingsHandler}
        >
          {renderSchedulerMenuItems()}
        </MenuOptionGroup>
      </MenuList>
    </Menu>
  );
}
