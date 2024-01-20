import { useDisclosure } from '@invoke-ai/ui';
import { useAppDispatch } from 'app/store/storeHooks';
import { expanderToggled } from 'features/settingsAccordions/store/actions';
import { useCallback } from 'react';

type UseExpanderToggleArg = {
  defaultIsOpen: boolean;
  id?: string;
};

export const useExpanderToggle = (arg: UseExpanderToggleArg) => {
  const dispatch = useAppDispatch();
  const { isOpen, onToggle: _onToggle } = useDisclosure({
    defaultIsOpen: arg.defaultIsOpen,
  });
  const onToggle = useCallback(() => {
    if (arg.id) {
      dispatch(expanderToggled({ id: arg.id, isOpen }));
    }
    _onToggle();
  }, [_onToggle, dispatch, arg.id, isOpen]);
  return { isOpen, onToggle };
};
