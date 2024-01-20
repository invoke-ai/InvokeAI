import { useDisclosure } from '@invoke-ai/ui';
import { useAppDispatch } from 'app/store/storeHooks';
import { standaloneAccordionToggled } from 'features/settingsAccordions/store/actions';
import { useCallback } from 'react';

type UseStandaloneAccordionToggleArg = {
  defaultIsOpen: boolean;
  id?: string;
};

export const useStandaloneAccordionToggle = (
  arg: UseStandaloneAccordionToggleArg
) => {
  const dispatch = useAppDispatch();
  const { isOpen, onToggle: _onToggle } = useDisclosure({
    defaultIsOpen: arg.defaultIsOpen,
  });
  const onToggle = useCallback(() => {
    if (arg.id) {
      dispatch(standaloneAccordionToggled({ id: arg.id, isOpen }));
    }
    _onToggle();
  }, [_onToggle, arg.id, dispatch, isOpen]);
  return { isOpen, onToggle };
};
