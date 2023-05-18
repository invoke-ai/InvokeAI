import {
  Box,
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
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { useTranslation } from 'react-i18next';

export default function SettingsSchedulers() {
  const schedulers = useAppSelector((state: RootState) => state.ui.schedulers);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const schedulerSettingsHandler = (v: string | string[]) => {
    if (isArray(v)) dispatch(setSchedulers(v.sort()));
  };

  return (
    <Menu closeOnSelect={false}>
      <MenuButton as={IAIButton}>
        {t('settings.availableSchedulers')}
      </MenuButton>
      <MenuList maxHeight={64} overflowY="scroll">
        <MenuOptionGroup
          value={schedulers}
          type="checkbox"
          onChange={schedulerSettingsHandler}
        >
          {SCHEDULERS.map((scheduler) => (
            <MenuItemOption key={scheduler} value={scheduler}>
              {scheduler}
            </MenuItemOption>
          ))}
        </MenuOptionGroup>
      </MenuList>
    </Menu>
  );
}
