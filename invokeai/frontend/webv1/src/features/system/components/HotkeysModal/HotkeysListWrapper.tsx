import type { SystemStyleObject } from '@invoke-ai/ui-library';
import type { AppThunkDispatch } from 'app/store/store';
import type { Hotkey, HotkeyConflictInfo } from 'features/system/components/HotkeysModal/useHotkeyData';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import type { TFunction } from 'i18next';
import { memo } from 'react';

import { HotkeyListItem } from './HotkeyListItem';

const HotkeyListWrapperContentSx: SystemStyleObject = {
  gap: 0,
  py: 0,
};

const HotkeyListWrapperHeadingSx: SystemStyleObject = {
  py: 3,
};

type HotkeysListWrapperProps = {
  title: string;
  hotkeysList: Hotkey[];
  conflictMap: Map<string, HotkeyConflictInfo>;
  t: TFunction;
  dispatch: AppThunkDispatch;
};

export const HotkeysListWrapper = memo((props: HotkeysListWrapperProps) => {
  const { title, hotkeysList, conflictMap, t, dispatch } = props;

  if (hotkeysList.length === 0) {
    return null;
  }

  return (
    <StickyScrollable title={title} headingSx={HotkeyListWrapperHeadingSx} contentSx={HotkeyListWrapperContentSx}>
      {hotkeysList.map((hotkey, index) => (
        <HotkeyListItem
          key={hotkey.id}
          lastItem={index === hotkeysList.length - 1}
          hotkey={hotkey}
          conflictMap={conflictMap}
          t={t}
          dispatch={dispatch}
        />
      ))}
    </StickyScrollable>
  );
});

HotkeysListWrapper.displayName = 'HotkeysListWrapper';
