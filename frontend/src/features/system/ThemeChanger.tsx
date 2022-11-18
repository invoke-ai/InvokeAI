import { useColorMode } from '@chakra-ui/react';
import React, { ChangeEvent } from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import IAISelect from '../../common/components/IAISelect';
import { setCurrentTheme } from '../options/optionsSlice';

const THEMES = ['dark', 'light', 'green'];

export default function ThemeChanger() {
  const { setColorMode } = useColorMode();
  const dispatch = useAppDispatch();
  const currentTheme = useAppSelector(
    (state: RootState) => state.options.currentTheme
  );

  const themeChangeHandler = (e: ChangeEvent<HTMLSelectElement>) => {
    setColorMode(e.target.value);
    dispatch(setCurrentTheme(e.target.value));
  };

  return (
    <IAISelect
      validValues={THEMES}
      value={currentTheme}
      onChange={themeChangeHandler}
      styleClass="theme-changer-dropdown"
    ></IAISelect>
  );
}
