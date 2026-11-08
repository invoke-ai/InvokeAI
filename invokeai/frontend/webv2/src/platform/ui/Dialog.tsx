// eslint-disable-next-line no-restricted-imports -- the one wrapper the restriction routes every dialog through
import { Dialog as ChakraDialog, useDialogContext } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';

import { registerModalPresence } from './modalPresence';

const ModalPresenceRegistration = () => {
  useMountEffect(registerModalPresence);

  return null;
};

// Follows the dialog's own open state rather than its DOM: a closing dialog keeps its content mounted, inert, through
// the exit animation, but stops being present the moment it closes, and is present again if it reopens meanwhile.
const ModalPresence = () => (useDialogContext().open ? <ModalPresenceRegistration /> : null);

const DialogRoot = ({ children, ...props }: ChakraDialog.RootProps) => (
  <ChakraDialog.Root {...props}>
    {props.modal === false ? null : <ModalPresence />}
    {children}
  </ChakraDialog.Root>
);

/**
 * Chakra's dialog parts with a root that announces modal presence while open, so application shortcuts behind it
 * stay inactive without each dialog registering itself. `RootProvider` is left out because it bypasses that root.
 */
export const Dialog = {
  ActionTrigger: ChakraDialog.ActionTrigger,
  Backdrop: ChakraDialog.Backdrop,
  Body: ChakraDialog.Body,
  CloseTrigger: ChakraDialog.CloseTrigger,
  Content: ChakraDialog.Content,
  Context: ChakraDialog.Context,
  Description: ChakraDialog.Description,
  Footer: ChakraDialog.Footer,
  Header: ChakraDialog.Header,
  /**
   * The suspense fallback of a dialog whose module is still loading. A root cannot stand in for it: the focus trap is
   * armed once, as the root opens, against content that would not exist yet.
   */
  Pending: ModalPresenceRegistration,
  Positioner: ChakraDialog.Positioner,
  Root: DialogRoot,
  Title: ChakraDialog.Title,
  Trigger: ChakraDialog.Trigger,
};

export declare namespace Dialog {
  export type OpenChangeDetails = ChakraDialog.OpenChangeDetails;
  export type RootProps = ChakraDialog.RootProps;
}
